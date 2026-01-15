# This is the replication stage

import tensorflow as tf # We will use TensorFlow for the text processing
import pandas as pd
import pandas as pd # Pandas implementation
from sklearn.model_selection import train_test_split # Train-Test-Split
from sklearn.preprocessing import StandardScaler # Standard Scaler
from sklearn.ensemble import RandomForestRegressor # Random Forest Regressor
from sklearn.linear_model import Ridge # Ridge Regression
from sklearn.metrics import mean_squared_error # MSE for simple accuracy
import numpy as np
from tensorflow.keras import layers, models


from tensorflow.keras.layers import TextVectorization # Text vectorization layer
from sklearn.model_selection import train_test_split # Train-test split

# Load the curated hybrid dataset
df = pd.read_csv("hfrp_dataset.csv")

# Define candidate numeric columns
candidate_num_cols = [  # NEW
    "revenue",          # NEW
    "net_income",       # NEW
    "operating_income", # NEW
    "eps",              # NEW
    "total_assets",     # NEW
    "total_liabilities",# NEW
    "cfo",              # NEW
]                       # NEW

# Keep only numeric columns that actually exist and aren't all NaN
num_cols = []
for c in candidate_num_cols:
    if c in df.columns and not df[c].isna().all():
        num_cols.append(c)

print("Using numeric columns:", num_cols)

# Drop rows with missing text, label, or numeric features; simple preprocessing steps
df = df.dropna(subset=["textual_disclosures", "future_vol"])
df = df.dropna(subset=num_cols)

print("Final training dataset shape after drops:", df.shape)

# Build arrays for model input
text_data = df["textual_disclosures"].astype(str).values
numeric_data = df[num_cols].values.astype("float32")
labels = df["future_vol"].values.astype("float32")


X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    text_data, numeric_data, labels, test_size=0.2, random_state=42
)

# Scale numeric features
scaler = StandardScaler()
X_num_train_scaled = scaler.fit_transform(X_num_train)
X_num_test_scaled = scaler.transform(X_num_test)

# Text vectorizer
max_tokens = 20000
max_len = 400 # Maximum length of text sequences

text_vec = TextVectorization( 
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_len,
)
text_vec.adapt(X_text_train)

vocab_size = text_vec.vocabulary_size()

# Essentially the code above is setting up the data for a hybrid model that uses both text and numeric features to predict future volatility.
# It is the prepare the data stage

# Next stage is the build the HFRP model stage
# We closely follow the architecture described in the HFRP paper

# Text input & branch; first we define the text input branch that processes the textual disclosures
text_input = layers.Input(shape=(1,), dtype=tf.string, name="text_input") # input layer for text data
x = text_vec(text_input) # vectorize the text input
x = layers.Embedding(input_dim=vocab_size, # Embedding layer to convert tokens to dense vectors
                     output_dim=128,   # embedding dimension
                     name="text_embedding")(x) # So far so good here; just embedding layer for deep learning architecture
x = layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu")(x) # the heart of this part; the CNN is implemented here as per the paper
x = layers.MaxPooling1D(pool_size=2)(x) # We can perform max pooling to reduce dimensionality as well; laerned about this in class
x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x) # BiLSTM layer to capture sequential dependencies in text; could be useful but LSTM will be more useful for the numerical aspect
x = layers.Dropout(0.3)(x) # Dropout for regularization; good sanity check
text_repr = layers.Dense(64, activation="relu")(x) # Relu activation function to introduce non-linearity

# Numeric input & branch; now we define the numeric input branch that processes the financial metrics; this uses the LSTMed numerical data
# We use multi-layer perceptron (MLP) for numeric data as per the paper; although no LSTM here since numeric data is not sequential in nature, but it shouldn't matter since we LSTM the text data
num_input = layers.Input(shape=(len(num_cols),), dtype="float32", name="num_input")
y_num = layers.Dense(32, activation="relu")(num_input)
y_num = layers.Dropout(0.2)(y_num)
num_repr = layers.Dense(32, activation="relu")(y_num)

# As the paper also does, we combine the text and numeric representations
combined = layers.concatenate([text_repr, num_repr]) # concatenate the two representations
z = layers.Dense(64, activation="relu")(combined) # fully connected layer after concatenation
z = layers.Dropout(0.3)(z) # dropout for regularization
output = layers.Dense(1, name="risk_output")(z)  # regression output for future volatility prediction

model = models.Model(inputs=[text_input, num_input], outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), # we do all the classic steps of training;
    # we already did data splitting and scaling above, and now we perform parameter optimization via Adam optimizer
    # Down here we also do validation and evaluation via RMSE metric
    loss="mse",
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
)

model.summary()


# Train the HFRP model
# and evaluate its performance

batch_size = 16
epochs = 10 # You can increase epochs for better performance; kept low for quick testing

history = model.fit(
    x={"text_input": X_text_train, "num_input": X_num_train_scaled},
    y=y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
)

test_metrics = model.evaluate(
    {"text_input": X_text_test, "num_input": X_num_test_scaled},
    y_test,
    verbose=0,
)
print("Test loss (MSE), RMSE:", test_metrics)
