import os
import pandas

from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

# --------------- Construct the structure of our model
cities_model = BayesianModel()
cities_model.add_nodes_from(['True NY', 'True CA', 'Predicted NY', 'Predicted CA'])
# cities_model.add_edges_from([('True NY', 'True CA'), ('True NY', 'Predicted NY'), \
# 	('True CA', 'Predicted CA')])
cities_model.add_edges_from([('True CA', 'True NY'), ('True NY', 'Predicted NY'), \
	('True CA', 'Predicted CA')])

# ---------------- Construct our data set 

true_data_trump_ca = [1] * 7; 
predicted_data_trump_ca = [1] * 7;

true_data_trump_ny = [1] * 7;
predicted_data_trump_ny = [0] * 2 + [1] * 5;

true_data_anaheim_ca = [1] * 6;
predicted_data_anaheim_ca = [0] * 1 + [1] * 5;

true_data_anaheim_ny = [0] * 6; 
predicted_data_anaheim_ny = [0] * 6;

true_data_flatbush_ca = [0] * 51;
predicted_data_flatbush_ca = [0] * 51;

true_data_flatbush_ny = [1] * 51;
predicted_data_flatbush_ny = [0] * (51-27) + [1] * 27;

true_data_garner_ca = [1] * 5
predicted_data_garner_ca = [1] * 5

true_data_garner_ny = [1] * 5
predicted_data_garner_ny = [1] * 3 + [0] * 2

true_data_occupy_ny = [1] * 6
predicted_data_occupy_ny = [1] * 4 + [0] * 2

true_data_occupy_ca = [1] * 6
predicted_data_occupy_ca = [0] * 4 + [1] * 2

true_data_alton_ny = [1] * 3
predicted_data_alton_ny = [1] * 2 + [0]

true_data_alton_ca = [0] * 3
predicted_data_alton_ca = [0] * 3

true_data_normal_ca = [0] * 99
predicted_data_normal_ca = [0] * 99

true_data_normal_ny = [0] * 99;
predicted_data_normal_ny = [0] * 99;

true_ca_labels = true_data_flatbush_ca + true_data_normal_ca + true_data_anaheim_ca + true_data_trump_ca + true_data_garner_ca + true_data_alton_ca
true_ny_labels = true_data_flatbush_ny + true_data_normal_ny + true_data_anaheim_ny + true_data_trump_ny + true_data_garner_ny + true_data_alton_ny

predicted_ca_labels = predicted_data_flatbush_ca + predicted_data_normal_ca + predicted_data_anaheim_ca + predicted_data_trump_ca + predicted_data_garner_ca + predicted_data_alton_ca
predicted_ny_labels = predicted_data_flatbush_ny + predicted_data_normal_ny + predicted_data_anaheim_ny + predicted_data_trump_ny + predicted_data_garner_ny + predicted_data_alton_ny

data = pandas.DataFrame(data={'True NY':true_ny_labels, 'True CA':true_ca_labels,\
	'Predicted NY':predicted_ny_labels, 'Predicted CA': predicted_ca_labels})

# Learn
cities_model.fit(data)
# print cities_model.get_cpds('Predicted CA')

# Test
# loop through all tuples of "Occupy data" and then average the probabilities
def testCalifornia(true_ny_data, true_ca_data, predicted_ca_data, predicted_ny_data):
	ca_infer = VariableElimination(cities_model)

	avg_prob = 0.0

	# Computing the probability of bronc given smoke.
	for index, true_ny_label in enumerate(true_ny_data):
		print "True NY label " + str(true_ny_label)
		true_ca_label = true_ca_data[index]
		print "True CA label " + str(true_ca_label)

		predicted_ca_label = predicted_ca_data[index]
		print "Predicted CA label " + str(predicted_ca_label)

		predicted_ny_label = predicted_ny_data[index]
		prob_ca_protest = ca_infer.query(variables=['True CA'], evidence={'True NY': true_ny_label, 'Predicted CA': predicted_ca_label, 'Predicted NY': predicted_ny_label})
		factor = prob_ca_protest['True CA']
		print factor
		# print factor.assignment([1])
		# avg_prob += prob_ca_protest

	# avg_prob /= len(true_ny_data)
	print avg_prob

testCalifornia(true_data_occupy_ny, true_data_occupy_ca, predicted_data_occupy_ca, predicted_data_occupy_ny)
