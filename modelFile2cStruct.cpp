#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <locale.h>
#include "svm.h"

struct svm_model* model;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static const char *svm_type_table[] =
{
	"c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

static const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};


static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}


//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var) do{ if (fscanf(_stream, _format, _var) != 1) return false; }while(0)
bool read_model_header(FILE *fp, svm_model* model)
{
	svm_parameter& param = model->param;
	char cmd[81];
	while(1)
	{
		FSCANF(fp,"%80s",cmd);

		if(strcmp(cmd,"svm_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;svm_type_table[i];i++)
			{
				if(strcmp(svm_type_table[i],cmd)==0)
				{
					param.svm_type=i;
					break;
				}
			}
			if(svm_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown svm type.\n");
				return false;
			}
		}
		else if(strcmp(cmd,"kernel_type")==0)
		{		
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;kernel_type_table[i];i++)
			{
				if(strcmp(kernel_type_table[i],cmd)==0)
				{
					param.kernel_type=i;
					break;
				}
			}
			if(kernel_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown kernel function.\n");	
				return false;
			}
		}
		else if(strcmp(cmd,"degree")==0)
			FSCANF(fp,"%d",&param.degree);
		else if(strcmp(cmd,"gamma")==0)
			FSCANF(fp,"%lf",&param.gamma);
		else if(strcmp(cmd,"coef0")==0)
			FSCANF(fp,"%lf",&param.coef0);
		else if(strcmp(cmd,"nr_class")==0)
			FSCANF(fp,"%d",&model->nr_class);
		else if(strcmp(cmd,"total_sv")==0)
			FSCANF(fp,"%d",&model->l);
		else if(strcmp(cmd,"rho")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->rho = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->rho[i]);
		}
		else if(strcmp(cmd,"label")==0)
		{
			int n = model->nr_class;
			model->label = Malloc(int,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%d",&model->label[i]);
		}
		else if(strcmp(cmd,"probA")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probA = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->probA[i]);
		}
		else if(strcmp(cmd,"probB")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probB = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->probB[i]);
		}
		else if(strcmp(cmd,"nr_sv")==0)
		{
			int n = model->nr_class;
			model->nSV = Malloc(int,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%d",&model->nSV[i]);
		}
		else if(strcmp(cmd,"SV")==0)
		{
			while(1)
			{
				int c = getc(fp);
				if(c==EOF || c=='\n') break;
			}
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			return false;
		}
	}

	return true;

}

svm_model *svm_load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"rb");
	if(fp==NULL) return NULL;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale) {
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	// read parameters

	svm_model *model = Malloc(svm_model,1);
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->sv_indices = NULL;
	model->label = NULL;
	model->nSV = NULL;
	
	// read header
	if (!read_model_header(fp, model))
	{
		fprintf(stderr, "ERROR: fscanf failed to read model\n");
		setlocale(LC_ALL, old_locale);
		free(old_locale);
		free(model->rho);
		free(model->label);
		free(model->nSV);
		free(model);
		return NULL;
	}
	
	// read sv_coef and SV

	int elements = 0;
	long pos = ftell(fp);

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	char *p,*endptr,*idx,*val;

	while(readline(fp)!=NULL)
	{
		p = strtok(line,":");
		while(1)
		{
			p = strtok(NULL,":");
			if(p == NULL)
				break;
			++elements;
		}
	}
	elements += model->l;

	fseek(fp,pos,SEEK_SET);

	int m = model->nr_class - 1;
	int l = model->l;
	model->sv_coef = Malloc(double *,m);
	int i;
	for(i=0;i<m;i++)
		model->sv_coef[i] = Malloc(double,l);
	model->SV = Malloc(svm_node*,l);
	svm_node *x_space = NULL;
	if(l>0) x_space = Malloc(svm_node,elements);

	int j=0;
	for(i=0;i<l;i++)
	{
		readline(fp);
		model->SV[i] = &x_space[j];

		p = strtok(line, " \t");
		model->sv_coef[0][i] = strtod(p,&endptr);
		for(int k=1;k<m;k++)
		{
			p = strtok(NULL, " \t");
			model->sv_coef[k][i] = strtod(p,&endptr);
		}

		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
				break;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			x_space[j].value = strtod(val,&endptr);

			++j;
		}
		x_space[j++].index = -1;
	}
	free(line);

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0)
		return NULL;

	model->free_sv = 1;	// XXX
	return model;
}

void svm_free_model_content(svm_model* model_ptr)
{
	if(model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)
		free((void *)(model_ptr->SV[0]));
	if(model_ptr->sv_coef)
	{
		for(int i=0;i<model_ptr->nr_class-1;i++)
			free(model_ptr->sv_coef[i]);
	}

	free(model_ptr->SV);
	model_ptr->SV = NULL;

	free(model_ptr->sv_coef);
	model_ptr->sv_coef = NULL;

	free(model_ptr->rho);
	model_ptr->rho = NULL;

	free(model_ptr->label);
	model_ptr->label= NULL;

	free(model_ptr->probA);
	model_ptr->probA = NULL;

	free(model_ptr->probB);
	model_ptr->probB= NULL;

	free(model_ptr->sv_indices);
	model_ptr->sv_indices = NULL;

	free(model_ptr->nSV);
	model_ptr->nSV = NULL;
}

void svm_free_and_destroy_model(svm_model** model_ptr_ptr)
{
	if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
	{
		svm_free_model_content(*model_ptr_ptr);
		free(*model_ptr_ptr);
		*model_ptr_ptr = NULL;
	}
}

void exit_with_help()
{
	printf(
	"Usage: modelFile2cStruct model_file > output_file\n"
	);
	exit(1);
}

int main(int argc, char **argv)
{

	if(argc != 2)
		exit_with_help();

	if((model=svm_load_model(argv[1]))==0)
	{
		fprintf(stderr,"can't open model file %s\n",argv[1]);
		exit(1);
	}


	// ---------- Declare for  struct svm_node **SV; -------------- //
	printf("struct svm_node SV_nodes[] = {\n");
	int l=0, k=0;
	for (;l<(model->l);l++) {
		k = 0;
		while (1) { 
			printf("\t{.index=%d, .value=%lf}, ", (model->SV)[l][k].index, (model->SV)[l][k].value);
			if ((model->SV)[l][k].index == -1) {
				printf("\n");
				break;
			}
			k++;
		}
	}
	printf("};\n");

	printf("struct svm_node *SV[%d] = {\n", model->l);
	for (l = 0; l<model->l; l++) {
		k = 0;
		while (1) { 					
			if ((model->SV)[l][k].index == -1) {
				printf("\t&SV_nodes[%d],\n", (k+1)*l);
				break;
			}
			k++;
		}	
	}
	printf("};\n");
	// ------------------------------------------------------------- //


	// ---------- Declare for double **sv_coef;	 -------------- //
	printf("double sv_coefs[%d][%d] = {\n", (model->nr_class) - 1, model->l);
	for (k = 0; k < (model->nr_class) - 1; k++) {
		printf("\t{");
		for (l = 0; l < (model->l); l++) {
			printf("%.16g, ", model->sv_coef[k][l]);
		}
		printf("},\n");
	}
	printf("};\n");

	printf("double *sv_coef[%d] = {\n", (model->nr_class) - 1);
	for (k = 0; k < (model->nr_class) - 1; k++) {
		printf("\t&sv_coefs[%d][0],\n",k);
	}
	printf("};\n");
	// ------------------------------------------------------------- //
	

	// ------------------- Declare for double *rho;	 -------------- //	
	int nr = (model->nr_class) * (model->nr_class - 1) / 2;
	printf("double rho[%d] = {\n", nr);
	for (k = 0; k < nr; k++) {
		printf("\t%g, \n", model->rho[k]);
	}
	printf("};\n\n");
	// ------------------------------------------------------------- //

	// ------------------- Declare for int *label;	 -------------- //	
	printf("int label[%d] = {\n", (model->nr_class));
	for (k = 0; k < (model->nr_class); k++) {
		printf("\t%d, \n", model->label[k]);
	}
	printf("};\n\n");
	// ------------------------------------------------------------- //


	// ------------------- Declare for int *nSV;	 -------------- //	
	printf("int nSV[%d] = {\n", (model->nr_class));
	for (k = 0; k < (model->nr_class); k++) {
		printf("\t%d, \n", model->nSV[k]);
	}
	printf("};\n\n");
	// ------------------------------------------------------------- //
	
	// ------ Declare for struct svm_parameter param;	 ----------- //
	// printf("struct svm_parameter param = { \n");
	// printf("\t.svm_type=%d, .kernel_type=%d, .degree=%d, .gamma=%g, .coef0=%g\n", model->param.svm_type, model->param.kernel_type, model->param.degree, model->param.gamma, model->param.coef0);
	// printf("}; \n");
	// ------------------------------------------------------------- //

	// --------------- Declare for struct svm_model;	 ----------- // 
	printf("struct svm_model svmModel = { \n");
	printf("\t{%d, %d, %d, %g, %g},\n", model->param.svm_type, model->param.kernel_type, model->param.degree, model->param.gamma, model->param.coef0);
	printf("\t.nr_class=%d,\n", model->nr_class);
	printf("\t.l=%d,\n", model->l);
	printf("\t.SV=SV,\n");
	printf("\t.sv_coef=sv_coef,\n");
	printf("\t.rho=rho,\n");
	printf("\t.label=label,\n");
	printf("\t.nSV=nSV,\n");
	printf("\t.free_sv=1\n");
	printf("};\n");
	// ------------------------------------------------------------- //


	svm_free_and_destroy_model(&model);
 
	return 0;
}