def evaluate_model(model, X_test, y_test):
    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=32)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

