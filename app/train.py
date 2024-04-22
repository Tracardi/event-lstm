from app.ml.train import train_model

# Data: event sequences and the next event
data = [
    'visit-started', 'page-view', 'page-view', 'contact', 'download', 'sign-in', 'visit-ended',
    'visit-started', 'page-view', 'page-view', 'page-view', 'page-view', 'page-view', 'download', 'sign-in',
    'page-view', 'page-view', 'page-view', 'visit-ended']

train_model(data, num_epochs=500)
