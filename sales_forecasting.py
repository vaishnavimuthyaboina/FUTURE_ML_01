import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("sales_data.csv")

# Create Day column
data["Day"] = range(1, len(data) + 1)

# Features and Target
X = data[["Day"]]
y = data["Sales"]

# Train Model
model = LinearRegression()
model.fit(X, y)

# Forecast next 30 days
future_days = pd.DataFrame({
    "Day": range(len(data) + 1, len(data) + 31)
})

predicted_sales = model.predict(future_days)

# Create forecast table
forecast = pd.DataFrame({
    "Day": future_days["Day"],
    "Predicted_Sales": predicted_sales
})

print("\nFuture Sales Forecast:")
print(forecast)

# Plot graph
plt.figure(figsize=(10, 5))
plt.plot(data["Day"], data["Sales"], marker="o", label="Historical Sales")
plt.plot(future_days["Day"], predicted_sales, marker="o", label="Forecasted Sales")

plt.xlabel("Day")
plt.ylabel("Sales")
plt.title("Sales & Demand Forecasting")
plt.legend()
plt.grid(True)

plt.show()