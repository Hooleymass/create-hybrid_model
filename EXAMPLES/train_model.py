def train_model(model, X_train, y_train):
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fit the model to the training data
    model.fit(X_train, y_train, epochs=5, batch_size=32)

