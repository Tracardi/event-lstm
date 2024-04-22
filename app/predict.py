# Test the model to predict 3 events ahead
from app.ml.predict import predict_next_events

length_of_predicted_sequence = 5
test_sequence = ['page-view', 'contact']
predicted_events, prediction_probabilities = predict_next_events(test_sequence, length_of_predicted_sequence)
print(f'Predicted next {length_of_predicted_sequence} events after {test_sequence} are {predicted_events} with probabilities {prediction_probabilities}')
