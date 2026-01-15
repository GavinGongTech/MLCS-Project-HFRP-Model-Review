# HFRP_architecture_experiments.py
# We compare several hybrid architectures for the HFRP task; this is my original research section

# This is the original research section

import tensorflow as tf # Deep learning framework (Keras lives inside this)
import pandas as pd # classic pandas
import numpy as np # classic numpy

from tensorflow.keras import layers, models # these are keras building blocks that we should use
from tensorflow.keras.layers import TextVectorization # a layer to convert raw text into integer token sequences
from sklearn.model_selection import train_test_split # train-test-split function
from sklearn.preprocessing import StandardScaler # standard scaler for numeric features
from sklearn.metrics import mean_squared_error # MSE metric for regression evaluation

TEXT_COL = "textual_disclosures" # column name for text data
NUM_COLS = ["revenue", "net_income", "operating_income", "eps", "total_assets"] # numeric feature columns

df = pd.read_csv("hfrp_dataset.csv") # Load the curated hybrid dataset
print("Loaded hfrp_dataset.csv with shape:", df.shape) # shape of dataset
print("Columns:", df.columns.tolist()) # print columns

# Keep only rows where we have text, numeric features, and target
df = df.dropna(subset=[TEXT_COL, "future_vol"] + NUM_COLS) # We drop rows with missing text, numeric features, or target
print("After dropping missing text/numerics/target:", df.shape) # print new shape after filtering

texts = df[TEXT_COL].astype(str).values # extract text data as a numpy array of strings
numeric_data = df[NUM_COLS].values.astype("float32") # extract numeric features as a numpy array of floats
labels = df["future_vol"].values.astype("float32") # extract target labels as a numpy array of floats

# Train/test split; this has to be fixed across all architectures for fair comparison
X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    texts, numeric_data, labels, test_size=0.2, random_state=42
)

# Scale numeric features
scaler = StandardScaler() # same steps as before
X_num_train_scaled = scaler.fit_transform(X_num_train)
X_num_test_scaled = scaler.transform(X_num_test)

# Text vectorizer which is shared across architectures
max_tokens = 20000
max_len = 400

text_vec = TextVectorization( # text vectorization layer
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_len,
)
text_vec.adapt(X_text_train) # fit the text vectorizer on training text data

vocab_size = text_vec.vocabulary_size()
print("Vocab size:", vocab_size)

# Here is where we define the different architectures to compare; this is the heart of our original research

def build_model(arch_name: str, vocab_size: int, num_features: int, text_vec_layer):
    # the arch_name selects which architecture to build; this is useful parameterization
    # Text branch
    text_input = layers.Input(shape=(1,), dtype=tf.string, name="text_input")
    x = text_vec_layer(text_input)
    x = layers.Embedding( # embedding layer to convert tokens to dense vectors
        input_dim=vocab_size,
        output_dim=128,
        name=f"text_embedding_{arch_name}" # unique name per architecture
    )(x)

    if arch_name == "cnn_bilstm": # this is our baseline architecture; the current one
        x = layers.Conv1D(filters=64, kernel_size=5, padding="same", # CNN layer with paramters of filters, kernel size, padding
                          activation="relu", name="conv1d")(x) # we also have relu activation here
        x = layers.MaxPooling1D(pool_size=2, name="max_pool")(x) # we pool the output of the conv layer via max pooling
        x = layers.Bidirectional( # bidirectional LSTM layer done here as well
            layers.LSTM(64, return_sequences=False), # we do not return sequences here; just the final output
            name="bilstm", # unique name; this is the next layer in the architecture
        )(x)
        x = layers.Dropout(0.3, name="dropout_text")(x) # simple dropout for regularization
        text_repr = layers.Dense(64, activation="relu", name="dense_text")(x) # final dense layer to get text representation

    elif arch_name == "bilstm_only": # only BiLSTM layers, no CNN
        x = layers.Bidirectional( # this is the first variant we test here; only a deeper biLSTM
            layers.LSTM(64, return_sequences=True), # first LSTM layer returns sequences
            name="bilstm1",
        )(x)
        x = layers.Bidirectional( # second LSTM layer that does not return sequences
            layers.LSTM(32, return_sequences=False),
            name="bilstm2",
        )(x) # note here that we stack two BiLSTM layers; this is deeper than the original
        x = layers.Dropout(0.3, name="dropout_text")(x) # dropout for regularization; like before
        text_repr = layers.Dense(64, activation="relu", name="dense_text")(x) # final dense layer for text representation

    elif arch_name == "cnn_only": # only CNN layers, no recurrent layers now
        # Pure CNN text encoder, no recurrent layers here
        x = layers.Conv1D(filters=64, kernel_size=5, padding="same", # same as before; conv1d layer with these parameters
                          activation="relu", name="conv1d_1")(x) # first conv1d layer
        x = layers.MaxPooling1D(pool_size=2, name="max_pool_1")(x) # max pooling layer
        x = layers.Conv1D(filters=64, kernel_size=5, padding="same", # second conv1d layer; this is deeper than original
                          activation="relu", name="conv1d_2")(x) # note two conv layers here
        x = layers.GlobalMaxPooling1D(name="global_max_pool")(x) # global max pooling to reduce dimensionality
        x = layers.Dropout(0.3, name="dropout_text")(x) # dropout for regularization
        text_repr = layers.Dense(64, activation="relu", name="dense_text")(x) # final dense layer for text representation

    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    # Numeric branch (same across architectures)
    num_input = layers.Input(shape=(num_features,), dtype="float32", name="num_input") # the numeric branch is the same across the architectures; nothing is changed here
    y_num = layers.Dense(32, activation="relu", name="dense_num1")(num_input) # first dense layer for numeric data
    y_num = layers.Dropout(0.2, name="dropout_num")(y_num) # dropout for regularization
    num_repr = layers.Dense(32, activation="relu", name="dense_num2")(y_num) # second dense layer for numeric data

    # Fusion + output; the fusion is between the identicla numeric branch and the different text branches
    combined = layers.concatenate([text_repr, num_repr], name="concat") # concatenate text and numeric representations
    z = layers.Dense(64, activation="relu", name="dense_fusion")(combined) # dense layer after concatenation
    z = layers.Dropout(0.3, name="dropout_fusion")(z) # dropout for regularization
    output = layers.Dense(1, name="risk_output")(z) # regression output for future volatility prediction

    model = models.Model(inputs=[text_input, num_input], outputs=output, name=arch_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), # once agian optimize with adam optimizer
        loss="mse", # record mse loss for regression
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")], # rmse metric for evaluation
    )
    return model

ARCHS = ["cnn_bilstm", "bilstm_only", "cnn_only"] # We now run the experiments for each architecture
results = {}

for arch in ARCHS:
    print("\n======================================")
    print(f"Training architecture: {arch}")
    print("======================================")

    model = build_model(arch, vocab_size, len(NUM_COLS), text_vec)
    model.summary()

    # Early stopping to avoid overfitting on tiny data; one of the key issues we covered in an earlier lecture 
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_rmse",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit( # train the model
        x={"text_input": X_text_train, "num_input": X_num_train_scaled},
        y=y_train,
        validation_split=0.25,
        batch_size=4,
        epochs=20,
        callbacks=[es],
    )

    # Evaluate
    test_loss, test_rmse = model.evaluate(
        {"text_input": X_text_test, "num_input": X_num_test_scaled},
        y_test,
        verbose=0,
    )

    preds = model.predict(
        {"text_input": X_text_test, "num_input": X_num_test_scaled},
        verbose=0,
    ).flatten()
    rmse_manual = np.sqrt(mean_squared_error(y_test, preds))

    print(f"{arch} - Test MSE:  {test_loss:.4f}")
    print(f"{arch} - Test RMSE (metric):  {test_rmse:.4f}")
    print(f"{arch} - Test RMSE (manual):  {rmse_manual:.4f}")

    results[arch] = float(test_rmse)

print("\n=== Architecture comparison summary (RMSE) ===")
for arch, rmse in results.items():
    print(f"{arch:12s} : {rmse:.4f}")
