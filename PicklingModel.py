import pickle
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Pickling a model

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

    
# Loading a model    

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
