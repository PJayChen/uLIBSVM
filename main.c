#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "svm.h"

/* ---------------------------------------------- */
/* -----------  Feature scale functions  --------- */         

// parameters for scale
double lower=-1.0,upper=1.0;
double feature_max[] = {0, 83.246094, 64.804276, 4007.618591, 9451.746395, 281.21875, 240.90625};
double feature_min[] = {0, -25.507813, 2.979167, 14.210388, 16.903748, 6.078125, 49.640625};


struct svm_node scale_output(int index, double value)
{
	struct svm_node result;

	/* skip single-valued attribute */
	if(feature_max[index] == feature_min[index])
		return;

	if(value == feature_min[index])
		value = lower;
	else if(value == feature_max[index])
		value = upper;
	else
		value = lower + (upper-lower) * 
			(value-feature_min[index])/
			(feature_max[index]-feature_min[index]);

	if(value != 0)
	{
		// printf("%d:%g ",index, value);
		result.index = index;
		result.value = value;
	}

	return result;
}

void scale(struct svm_node *input_nodes)
{
	while ((input_nodes)->index != -1) {
		*input_nodes = scale_output(input_nodes->index, input_nodes->value);
		input_nodes++;
	}
}
/* ---------------------------------------------- */
/* ---------------------------------------------- */


/* ---------------------------------------------- */
/* ----------------  SVM Model ------------------ */  

struct svm_node SV_nodes[] = {
		{.index=1, .value=0.243705}, {.index=2, .value=-0.602541}, {.index=3, .value=-0.660792}, {.index=4, .value=-0.169692}, {.index=5, .value=-0.623715}, {.index=6, .value=-0.528470}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.418388}, {.index=2, .value=-0.485528}, {.index=3, .value=-0.803673}, {.index=4, .value=-0.672704}, {.index=5, .value=-0.548753}, {.index=6, .value=-0.180622}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.766172}, {.index=2, .value=-0.292106}, {.index=3, .value=-0.662158}, {.index=4, .value=-0.621895}, {.index=5, .value=-0.389290}, {.index=6, .value=-0.217874}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.247184}, {.index=2, .value=-0.505349}, {.index=3, .value=-0.767049}, {.index=4, .value=-0.529327}, {.index=5, .value=-0.574649}, {.index=6, .value=0.008414}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.919672}, {.index=2, .value=-0.336569}, {.index=3, .value=-0.645740}, {.index=4, .value=-0.720735}, {.index=5, .value=-0.436765}, {.index=6, .value=-0.065436}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=-0.275277}, {.index=2, .value=-0.512001}, {.index=3, .value=-0.887612}, {.index=4, .value=-0.465589}, {.index=5, .value=-0.649838}, {.index=6, .value=-0.332571}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=-0.317769}, {.index=2, .value=-0.389428}, {.index=3, .value=-0.876730}, {.index=4, .value=-0.595452}, {.index=5, .value=-0.536260}, {.index=6, .value=-0.549056}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.527819}, {.index=2, .value=-0.427821}, {.index=3, .value=-0.731896}, {.index=4, .value=-0.778834}, {.index=5, .value=-0.495485}, {.index=6, .value=0.337963}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.616752}, {.index=2, .value=-0.930247}, {.index=3, .value=-0.953376}, {.index=4, .value=-0.964082}, {.index=5, .value=-0.906639}, {.index=6, .value=-0.848705}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.545706}, {.index=2, .value=-0.821405}, {.index=3, .value=-0.979290}, {.index=4, .value=-0.977316}, {.index=5, .value=-0.941166}, {.index=6, .value=-0.862103}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.551022}, {.index=2, .value=-0.880038}, {.index=3, .value=-0.970590}, {.index=4, .value=-0.975920}, {.index=5, .value=-0.938327}, {.index=6, .value=-0.797402}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.648217}, {.index=2, .value=-0.956615}, {.index=3, .value=-0.964539}, {.index=4, .value=-0.979988}, {.index=5, .value=-0.994208}, {.index=6, .value=-0.840536}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.574800}, {.index=2, .value=-0.900593}, {.index=3, .value=-0.944792}, {.index=4, .value=-0.960665}, {.index=5, .value=-0.950593}, {.index=6, .value=-0.799853}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.621637}, {.index=2, .value=-0.932437}, {.index=3, .value=-0.946934}, {.index=4, .value=-0.971645}, {.index=5, .value=-0.964450}, {.index=6, .value=-0.826648}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.621733}, {.index=2, .value=-0.970346}, {.index=3, .value=-0.956257}, {.index=4, .value=-0.977583}, {.index=5, .value=-0.957295}, {.index=6, .value=-0.846745}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.702094}, {.index=2, .value=-0.824438}, {.index=3, .value=-0.943215}, {.index=4, .value=-0.963418}, {.index=5, .value=-0.871202}, {.index=6, .value=-0.750837}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.980368}, {.index=2, .value=-0.354650}, {.index=3, .value=-0.775327}, {.index=4, .value=-0.906711}, {.index=5, .value=-0.368732}, {.index=6, .value=0.004983}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.658489}, {.index=2, .value=-0.726758}, {.index=3, .value=-0.960105}, {.index=4, .value=-0.973374}, {.index=5, .value=-0.862684}, {.index=6, .value=-0.705253}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.646205}, {.index=2, .value=-0.822921}, {.index=3, .value=-0.999606}, {.index=4, .value=-0.996176}, {.index=5, .value=-0.956840}, {.index=6, .value=-0.803447}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.592998}, {.index=2, .value=-0.942715}, {.index=3, .value=-0.962001}, {.index=4, .value=-0.969868}, {.index=5, .value=-0.961838}, {.index=6, .value=-0.849522}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.634217}, {.index=2, .value=-0.730984}, {.index=3, .value=-0.956626}, {.index=4, .value=-0.964942}, {.index=5, .value=-0.768527}, {.index=6, .value=-0.750674}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.663973}, {.index=2, .value=-1.000000}, {.index=3, .value=-0.962220}, {.index=4, .value=-0.980152}, {.index=5, .value=-0.971605}, {.index=6, .value=-0.799199}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.764563}, {.index=2, .value=-0.608506}, {.index=3, .value=-0.923974}, {.index=4, .value=-0.979336}, {.index=5, .value=-0.696405}, {.index=6, .value=-0.528797}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.803355}, {.index=2, .value=-0.752764}, {.index=3, .value=-0.973197}, {.index=4, .value=-0.984790}, {.index=5, .value=-0.830996}, {.index=6, .value=-0.536639}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.703818}, {.index=2, .value=-0.805483}, {.index=3, .value=-0.990977}, {.index=4, .value=-0.981078}, {.index=5, .value=-0.953092}, {.index=6, .value=-0.761621}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.663015}, {.index=2, .value=-0.762699}, {.index=3, .value=-0.959026}, {.index=4, .value=-0.968220}, {.index=5, .value=-0.831563}, {.index=6, .value=-0.572747}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.664288}, {.index=2, .value=-0.798876}, {.index=3, .value=-0.973161}, {.index=4, .value=-0.977415}, {.index=5, .value=-0.854393}, {.index=6, .value=-0.755902}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.662276}, {.index=2, .value=-0.775336}, {.index=3, .value=-0.949424}, {.index=4, .value=-0.969133}, {.index=5, .value=-0.798058}, {.index=6, .value=-0.693326}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.666997}, {.index=2, .value=-0.751868}, {.index=3, .value=-0.965417}, {.index=4, .value=-0.965338}, {.index=5, .value=-0.849509}, {.index=6, .value=-0.700841}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.750727}, {.index=2, .value=-0.757717}, {.index=3, .value=-0.943016}, {.index=4, .value=-0.993101}, {.index=5, .value=-0.774661}, {.index=6, .value=-0.627645}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.500760}, {.index=2, .value=-0.518130}, {.index=3, .value=-0.768515}, {.index=4, .value=-0.876190}, {.index=5, .value=-0.548413}, {.index=6, .value=-0.267217}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.441644}, {.index=2, .value=-0.751146}, {.index=3, .value=-0.896941}, {.index=4, .value=-0.770579}, {.index=5, .value=-0.825884}, {.index=6, .value=-0.079160}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.677497}, {.index=2, .value=-0.507616}, {.index=3, .value=-0.670207}, {.index=4, .value=-0.788333}, {.index=5, .value=-0.579306}, {.index=6, .value=-0.585655}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.381991}, {.index=2, .value=-0.948275}, {.index=3, .value=-0.971499}, {.index=4, .value=-0.815124}, {.index=5, .value=-1.000000}, {.index=6, .value=-0.917654}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.399136}, {.index=2, .value=-0.922665}, {.index=3, .value=-0.970810}, {.index=4, .value=-0.882131}, {.index=5, .value=-0.959680}, {.index=6, .value=-0.817335}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.614166}, {.index=2, .value=-0.982814}, {.index=3, .value=-0.918659}, {.index=4, .value=-0.989366}, {.index=5, .value=-0.951275}, {.index=6, .value=-0.863083}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.589814}, {.index=2, .value=-0.780842}, {.index=3, .value=-0.899757}, {.index=4, .value=-0.931397}, {.index=5, .value=-0.783406}, {.index=6, .value=-0.738583}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.582127}, {.index=2, .value=-0.824185}, {.index=3, .value=-0.990985}, {.index=4, .value=-0.960283}, {.index=5, .value=-0.967517}, {.index=6, .value=-0.875500}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.555117}, {.index=2, .value=-0.874983}, {.index=3, .value=-0.981783}, {.index=4, .value=-0.981718}, {.index=5, .value=-0.970242}, {.index=6, .value=-0.871252}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.569628}, {.index=2, .value=-0.848110}, {.index=3, .value=-0.935532}, {.index=4, .value=-0.947073}, {.index=5, .value=-0.830541}, {.index=6, .value=-0.310514}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.608994}, {.index=2, .value=-0.846509}, {.index=3, .value=-0.965527}, {.index=4, .value=-0.939864}, {.index=5, .value=-0.936169}, {.index=6, .value=-0.762764}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.562731}, {.index=2, .value=-0.950549}, {.index=3, .value=-0.978312}, {.index=4, .value=-0.983631}, {.index=5, .value=-0.965245}, {.index=6, .value=-0.889388}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.594722}, {.index=2, .value=-0.649718}, {.index=3, .value=-0.896018}, {.index=4, .value=-0.974412}, {.index=5, .value=-0.781930}, {.index=6, .value=-0.756066}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.616369}, {.index=2, .value=-0.761929}, {.index=3, .value=-0.984593}, {.index=4, .value=-0.977073}, {.index=5, .value=-0.950253}, {.index=6, .value=-0.719141}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.586102}, {.index=2, .value=-0.677855}, {.index=3, .value=-0.740315}, {.index=4, .value=-0.949181}, {.index=5, .value=-0.648930}, {.index=6, .value=-0.485500}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.627240}, {.index=2, .value=-0.801566}, {.index=3, .value=-0.921093}, {.index=4, .value=-0.979416}, {.index=5, .value=-0.876995}, {.index=6, .value=-0.801323}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.652024}, {.index=2, .value=-0.898234}, {.index=3, .value=-0.971073}, {.index=4, .value=-0.977208}, {.index=5, .value=-0.901868}, {.index=6, .value=-0.743975}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.669337}, {.index=2, .value=-0.742890}, {.index=3, .value=-0.899411}, {.index=4, .value=-0.971361}, {.index=5, .value=-0.861548}, {.index=6, .value=-0.696920}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.623457}, {.index=2, .value=-0.784506}, {.index=3, .value=-0.972643}, {.index=4, .value=-0.974297}, {.index=5, .value=-0.924016}, {.index=6, .value=-0.733682}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.502389}, {.index=2, .value=-0.849458}, {.index=3, .value=-0.943924}, {.index=4, .value=-0.905755}, {.index=5, .value=-0.868931}, {.index=6, .value=-0.718487}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.658609}, {.index=2, .value=-0.945411}, {.index=3, .value=-0.947512}, {.index=4, .value=-0.975896}, {.index=5, .value=-0.918337}, {.index=6, .value=-0.783678}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.558134}, {.index=2, .value=-0.924139}, {.index=3, .value=-0.969527}, {.index=4, .value=-0.971934}, {.index=5, .value=-0.915384}, {.index=6, .value=-0.838085}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.636076}, {.index=2, .value=-0.825448}, {.index=3, .value=-0.980317}, {.index=4, .value=-0.994280}, {.index=5, .value=-0.904935}, {.index=6, .value=-0.690712}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.644816}, {.index=2, .value=-0.725789}, {.index=3, .value=-0.939042}, {.index=4, .value=-0.995005}, {.index=5, .value=-0.792720}, {.index=6, .value=-0.733192}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.627154}, {.index=2, .value=0.114805}, {.index=3, .value=0.236656}, {.index=4, .value=-0.961401}, {.index=5, .value=-0.029928}, {.index=6, .value=-0.780573}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.927445}, {.index=2, .value=0.467766}, {.index=3, .value=1.000000}, {.index=4, .value=-0.883803}, {.index=5, .value=0.349991}, {.index=6, .value=-0.208888}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.782040}, {.index=2, .value=-0.071080}, {.index=3, .value=-0.658142}, {.index=4, .value=-0.976777}, {.index=5, .value=-0.278437}, {.index=6, .value=-0.378972}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.659890}, {.index=2, .value=-0.093040}, {.index=3, .value=-0.344056}, {.index=4, .value=-0.923337}, {.index=5, .value=-0.230621}, {.index=6, .value=-0.379462}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.629060}, {.index=2, .value=-0.675327}, {.index=3, .value=-0.925986}, {.index=4, .value=-0.991558}, {.index=5, .value=-0.801124}, {.index=6, .value=-0.702965}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.677849}, {.index=2, .value=-0.722335}, {.index=3, .value=-0.943017}, {.index=4, .value=-0.991559}, {.index=5, .value=-0.760918}, {.index=6, .value=-0.745609}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.645343}, {.index=2, .value=-0.787350}, {.index=3, .value=-0.938116}, {.index=4, .value=-0.994967}, {.index=5, .value=-0.786473}, {.index=6, .value=-0.761948}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.640200}, {.index=2, .value=-0.784911}, {.index=3, .value=-0.913224}, {.index=4, .value=-0.999090}, {.index=5, .value=-0.799875}, {.index=6, .value=-0.816028}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.662498}, {.index=2, .value=-0.792189}, {.index=3, .value=-0.937501}, {.index=4, .value=-0.998041}, {.index=5, .value=-0.820660}, {.index=6, .value=-0.766849}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.238720}, {.index=2, .value=-0.649246}, {.index=3, .value=-0.876964}, {.index=4, .value=-0.298211}, {.index=5, .value=-0.771367}, {.index=6, .value=-0.345478}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.130084}, {.index=2, .value=-0.843645}, {.index=3, .value=-0.977836}, {.index=4, .value=-0.790234}, {.index=5, .value=-0.924357}, {.index=6, .value=-0.851156}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.289753}, {.index=2, .value=-0.764372}, {.index=3, .value=-0.858030}, {.index=4, .value=-0.593214}, {.index=5, .value=-0.710148}, {.index=6, .value=0.263622}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.587826}, {.index=2, .value=-0.972705}, {.index=3, .value=-0.974120}, {.index=4, .value=-0.930160}, {.index=5, .value=-0.948322}, {.index=6, .value=-0.779920}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.609090}, {.index=2, .value=-0.876163}, {.index=3, .value=-0.963514}, {.index=4, .value=-0.921454}, {.index=5, .value=-0.897780}, {.index=6, .value=-0.751001}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.640602}, {.index=2, .value=-0.880543}, {.index=3, .value=-0.976065}, {.index=4, .value=-0.939031}, {.index=5, .value=-0.938100}, {.index=6, .value=-0.816682}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.583995}, {.index=2, .value=-0.896381}, {.index=3, .value=-0.982841}, {.index=4, .value=-0.913407}, {.index=5, .value=-0.890851}, {.index=6, .value=-0.792991}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.676999}, {.index=2, .value=-0.979445}, {.index=3, .value=-0.955914}, {.index=4, .value=-0.978899}, {.index=5, .value=-0.956840}, {.index=6, .value=-0.780083}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.590125}, {.index=2, .value=-0.804219}, {.index=3, .value=-0.959023}, {.index=4, .value=-0.883976}, {.index=5, .value=-0.889148}, {.index=6, .value=-0.755902}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=-1.000000}, {.index=2, .value=-0.899245}, {.index=3, .value=-0.988020}, {.index=4, .value=-0.614451}, {.index=5, .value=-0.966154}, {.index=6, .value=-0.856874}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=-0.177568}, {.index=2, .value=-0.856450}, {.index=3, .value=-0.981839}, {.index=4, .value=-0.688854}, {.index=5, .value=-0.926969}, {.index=6, .value=-0.608202}, {.index=-1, .value=0.000000}, 
		{.index=1, .value=0.301438}, {.index=2, .value=-0.812307}, {.index=3, .value=-0.944757}, {.index=4, .value=-0.709179}, {.index=5, .value=-0.884945}, {.index=6, .value=-0.195981}, {.index=-1, .value=0.000000}, 
		};
struct svm_node *SV[75] = {
	&SV_nodes[0],
	&SV_nodes[7],
	&SV_nodes[14],
	&SV_nodes[21],
	&SV_nodes[28],
	&SV_nodes[35],
	&SV_nodes[42],
	&SV_nodes[49],
	&SV_nodes[56],
	&SV_nodes[63],
	&SV_nodes[70],
	&SV_nodes[77],
	&SV_nodes[84],
	&SV_nodes[91],
	&SV_nodes[98],
	&SV_nodes[105],
	&SV_nodes[112],
	&SV_nodes[119],
	&SV_nodes[126],
	&SV_nodes[133],
	&SV_nodes[140],
	&SV_nodes[147],
	&SV_nodes[154],
	&SV_nodes[161],
	&SV_nodes[168],
	&SV_nodes[175],
	&SV_nodes[182],
	&SV_nodes[189],
	&SV_nodes[196],
	&SV_nodes[203],
	&SV_nodes[210],
	&SV_nodes[217],
	&SV_nodes[224],
	&SV_nodes[231],
	&SV_nodes[238],
	&SV_nodes[245],
	&SV_nodes[252],
	&SV_nodes[259],
	&SV_nodes[266],
	&SV_nodes[273],
	&SV_nodes[280],
	&SV_nodes[287],
	&SV_nodes[294],
	&SV_nodes[301],
	&SV_nodes[308],
	&SV_nodes[315],
	&SV_nodes[322],
	&SV_nodes[329],
	&SV_nodes[336],
	&SV_nodes[343],
	&SV_nodes[350],
	&SV_nodes[357],
	&SV_nodes[364],
	&SV_nodes[371],
	&SV_nodes[378],
	&SV_nodes[385],
	&SV_nodes[392],
	&SV_nodes[399],
	&SV_nodes[406],
	&SV_nodes[413],
	&SV_nodes[420],
	&SV_nodes[427],
	&SV_nodes[434],
	&SV_nodes[441],
	&SV_nodes[448],
	&SV_nodes[455],
	&SV_nodes[462],
	&SV_nodes[469],
	&SV_nodes[476],
	&SV_nodes[483],
	&SV_nodes[490],
	&SV_nodes[497],
	&SV_nodes[504],
	&SV_nodes[511],
	&SV_nodes[518],
};

double sv_coefs[4][75] = {
	{0, 0, 0, 0, 503.8912110485852, 0, 0, 34.37080999697162, -0, -0, -0, -0, -27.31834144308781, -0, -0, -0, -510.943679602469, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -556.6452631931905, -106.9701731953055, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -46.97585891333769, -0, -0, -0, -75.93900600788089, -72.75361844570347, -0, -0, -0, -0, -0, -508.4440772475077, -0, -0, -0, -0, -0, -0, -0, -0, -30.19815583476035, -0, -0, },
	{0, 610.4515389433644, 0, 0, 53.16389744513163, 0, 0, 0, 8192, 8192, 8192, 5121.088373757932, 8192, 8192, 8192, 0, 0, 8192, 8192, 8192, 1876.862203392132, 0, 0, 0, 0, 7643.729747972126, 0, 0, 4800.693356355839, 0, -0, -0, -0, -0, -0, -8192, -0, -8192, -4007.33595980468, -1542.67923703818, -8192, -8192, -0, -8192, -0, -8192, -8192, -5700.358484635172, -8192, -0, -8192, -8192, -8192, -8192, -0, -0, -0, -0, -1284.865106190188, -8192, -1365.514361940837, -0, -8192, -0, -0, -0, -2216.583911473422, -8192, -8192, -0, -8192, -0, -0, -0, -0, },
	{0, 0, 115.7607210366768, 0, 52.68967997575673, 0, 27.21808235448854, 0, 987.3205971030022, 8192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8192, 0, 2702.956090852719, 0, 0, 0, 4134.210172589421, 3017.892607585882, 0, 8192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8192, 0, 363.7462081937384, 1717.465456788465, 8192, 4637.498781716889, 0, 0, 0, 0, -8192, -0, -90.38436409265401, -50.7366929046844, -0, -0, -5875.722755882656, -0, -0, -8192, -701.8666338190952, -0, -8192, -0, -8192, -8192, -8192, -1187.515924326015, -8192, -8192, -0, -234.982601620225, -991.7996424583423, },
	{289.6934697616232, 61.56710014852487, 0, 44.72408224613955, 0, 142.6575809259803, 0, 0, 7222.669642186235, 0, 0, 0, 5921.997962539835, 0, 0, 360.8273024822182, 0, 0, 0, 0, 0, 8192, 0, 117.0807181705334, 2128.036588598392, 2849.97169749621, 0, 0, 0, 0, 0, 1306.311421095646, 161.164175342941, 8192, 8192, 0, 642.454875998564, 2419.098626002839, 0, 0, 8192, 0, 0, 0, 0, 0, 8192, 0, 0, 8192, 6077.269069964598, 0, 2321.145391040832, 0, 0, 0, 0, 0, 0, 0, 0, 600.6335647475811, 0, -0, -0, -19.43053846416807, -0, -2593.855276461479, -0, -0, -0, -308.4931408627651, -0, -0, -0, },
};
double *sv_coef[4] = {
	&sv_coefs[0][0],
	&sv_coefs[1][0],
	&sv_coefs[2][0],
	&sv_coefs[3][0],
};

double rho[10] = {
	0.833603, 
	0.056457, 
	0.424754, 
	-0.587318, 
	-2.88788, 
	-22.4581, 
	-9.62578, 
	11.2043, 
	22.8494, 
	0.334467, 
};

int label[5] = {
	0, 
	3, 
	4, 
	1, 
	2, 
};

int nSV[5] = {
	8, 
	22, 
	22, 
	11, 
	12, 
};

struct svm_model svmModel = { 
	{0, 2, 0, 0.03125, 0},
	.nr_class=5,
	.l=75,
	.SV=SV,
	.sv_coef=sv_coef,
	.rho=rho,
	.label=label,
	.nSV=nSV,
	.free_sv=1
};
/* ---------------------------------------------- */
/* ---------------------------------------------- */

/* ---------------------------------------------- */
/* ----------  SVM Predict Function ------------- */  
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

double expT(double x) {
	double sum = 1.0f; // initialize sum of series
 	int i;
 	int n = 10;
    for (i = n - 1; i > 0; --i )
        sum = 1 + x * sum / i;
 
    return sum;
}

static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;
	int t;
	for(t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}

double dot(const struct svm_node *px, const struct svm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}

double k_function(const struct svm_node *x, const struct svm_node *y,
			  const struct svm_parameter *param)
{
	switch(param->kernel_type)
	{
		case LINEAR:
			return dot(x,y);
		case POLY:
			return powi(param->gamma*dot(x,y)+param->coef0,param->degree);
		case RBF:
		{
			double sum = 0;
			while(x->index != -1 && y->index !=-1)
			{
				if(x->index == y->index)
				{
					double d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else
				{
					if(x->index > y->index)
					{	
						sum += y->value * y->value;
						++y;
					}
					else
					{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += y->value * y->value;
				++y;
			}
			
			return expT(-param->gamma*sum);
		}
		case SIGMOID:
			return tanh(param->gamma*dot(x,y)+param->coef0);
		case PRECOMPUTED:  //x: test (validation), y: SV
			return x[(int)(y->value)].value;
		default:
			return 0;  // Unreachable 
	}
}

double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values)
{
	int i, j;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
	{
		double *sv_coef = model->sv_coef[0];
		double sum = 0;
		for(i=0;i<model->l;i++)
			sum += sv_coef[i] * k_function(x,model->SV[i], &(model->param));
		sum -= model->rho[0];
		*dec_values = sum;

		if(model->param.svm_type == ONE_CLASS)
			return (sum>0)?1:-1;
		else
			return sum;
	}
	else
	{
		int nr_class = model->nr_class;
		int l = model->l;
		
		double *kvalue = Malloc(double,l);
		for(i=0;i<l;i++)
			kvalue[i] = k_function(x, model->SV[i], &(model->param));

		int *start = Malloc(int,nr_class);
		start[0] = 0;
		for(i=1;i<nr_class;i++)
			start[i] = start[i-1]+model->nSV[i-1];

		int *vote = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			vote[i] = 0;

		int p=0;
		for(i=0;i<nr_class;i++)
			for(j=i+1;j<nr_class;j++)
			{
				double sum = 0;
				int si = start[i];
				int sj = start[j];
				int ci = model->nSV[i];
				int cj = model->nSV[j];
				
				int k;
				double *coef1 = model->sv_coef[j-1];
				double *coef2 = model->sv_coef[i];
				for(k=0;k<ci;k++)
					sum += coef1[si+k] * kvalue[si+k];
				for(k=0;k<cj;k++)
					sum += coef2[sj+k] * kvalue[sj+k];
				sum -= model->rho[p];
				dec_values[p] = sum;

				if(dec_values[p] > 0)
					++vote[i];
				else
					++vote[j];
				p++;
			}

		int vote_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(vote[i] > vote[vote_max_idx])
				vote_max_idx = i;

		free(kvalue);
		free(start);
		free(vote);
		return model->label[vote_max_idx];
	}
}

double svm_predict(const struct svm_model *model, const struct svm_node *x)
{
	int nr_class = model->nr_class;
	double *dec_values;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
		dec_values = Malloc(double, 1);
	else 
		dec_values = Malloc(double, nr_class*(nr_class-1)/2);
	double pred_result = svm_predict_values(model, x, dec_values);
	free(dec_values);
	return pred_result;
}
/* ---------------------------------------------- */
/* ---------------------------------------------- */





int main(void) {
	// target motion: 4
	// struct svm_node testingSet[7] = {
	// 	{1, 52.630208},
	// 	{2, 6.526042},
	// 	{3, 127.499322},
	// 	{4, 777.024712},
	// 	{5, 13.609375},
	// 	{6, 65.906250 },
	// 	{-1, 0.0}
	// };
	
	// target motion: 0
	// struct svm_node testingSet[7] = {
	// 	{1, 66.090625},
	// 	{2, 24.118750},
	// 	{3, 441.404648},
	// 	{4, 1227.713662},
	// 	{5, 80.906250},
	// 	{6, 118.890625},
	// 	{-1, 0.0}
	// };

	// target motion: 3
	struct svm_node testingSet[7] = {
		{1, 65.171875},
		{2, 11.845982},
		{3, 58.400386},
		{4, 106.236084},
		{5, 46.078125},
		{6, 110.843750},
		{-1, 0.0}
	};

	// target motion: 4
	// struct svm_node testingSet[7] = {
	// 	{1, 58.539063},
	// 	{2, 13.488281},
	// 	{3, 550.304550},
	// 	{4, 483.115662},
	// 	{5, 45.125000},
	// 	{6, 74.203125 },
	// 	{-1, 0.0}
	// };

	printf("\nTest feature vector:\n");
	int i;
	for (i=0;i<6;i++) {
		printf("%g\n", testingSet[i].value);
	}


	scale(testingSet);

	printf("\nScaled test feature vector:\n");
	for (i=0;i<6;i++) {
		printf("%g\n", testingSet[i].value);
	}

	printf("\nRecognition Result:\n");
	printf("%g\n", svm_predict(&svmModel, testingSet));


	printf("%g\n", powi((int)3,2));

}
