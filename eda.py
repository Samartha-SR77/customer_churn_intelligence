import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots



df = pd.read_csv("data/telco_churn_cleaned.csv")

os.makedirs("outputs", exist_ok=True)

print("Dataset shape:", df.shape)
print("\nChurn distribution:")
print(df["Churn"].value_counts())

overall_churn = df["Churn"].eq("Yes").mean() * 100
print(f"\nOverall churn rate: {overall_churn:.2f}%")



def churn_rate_by(column):
    """Calculate churn percentage within each category."""
    return (
        df.groupby(column, dropna=False)["Churn"]
        .apply(lambda s: s.eq("Yes").mean() * 100)
        .round(2)
        .sort_values(ascending=False)
    )



contract_rates = churn_rate_by("Contract")
payment_rates = churn_rate_by("PaymentMethod")
internet_rates = churn_rate_by("InternetService")
security_rates = churn_rate_by("OnlineSecurity")
support_rates = churn_rate_by("TechSupport")

senior_df = df.copy()
senior_df["CustomerGroup"] = senior_df["SeniorCitizen"].map(
    {0: "Non-senior", 1: "Senior"}
)
senior_rates = churn_rate_by("SeniorCitizen")
senior_rates.index = [
    "Non-senior" if value == 0 else "Senior"
    for value in senior_rates.index
]

print("\nChurn rate by contract (%):")
print(contract_rates)

print("\nChurn rate by payment method (%):")
print(payment_rates)

print("\nChurn rate by internet service (%):")
print(internet_rates)

print("\nChurn rate by senior-citizen status (%):")
print(senior_rates)

print("\nChurn rate by OnlineSecurity (%):")
print(security_rates)

print("\nChurn rate by TechSupport (%):")
print(support_rates)



print("\nAverage tenure by churn status:")
print(df.groupby("Churn")["tenure"].mean().round(2))

print("\nAverage monthly charges by churn status:")
print(df.groupby("Churn")["MonthlyCharges"].mean().round(2))



fig = make_subplots(
    rows=4,
    cols=2,
    subplot_titles=(
        "Overall Customer Churn",
        "Churn Rate by Contract",
        "Churn Rate by Payment Method",
        "Churn Rate by Internet Service",
        "Tenure by Churn Status",
        "Monthly Charges by Churn Status",
        "Churn Rate by Senior-Citizen Status",
        "Churn Rate: Security and Tech Support"
    ),
    specs=[
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "box"}, {"type": "box"}],
        [{"type": "bar"}, {"type": "bar"}]
    ],
    vertical_spacing=0.09,
    horizontal_spacing=0.12
)


churn_counts = df["Churn"].value_counts().reindex(
    ["No", "Yes"], fill_value=0
)

fig.add_trace(
    go.Bar(
        x=["Stayed", "Churned"],
        y=churn_counts.values,
        text=churn_counts.values,
        textposition="outside",
        name="Customers",
        showlegend=False,
        hovertemplate="%{x}: %{y} customers<extra></extra>"
    ),
    row=1, col=1
)


fig.add_trace(
    go.Bar(
        x=contract_rates.index,
        y=contract_rates.values,
        text=[f"{v:.2f}%" for v in contract_rates.values],
        textposition="outside",
        name="Contract",
        showlegend=False,
        hovertemplate="%{x}<br>Churn rate: %{y:.2f}%<extra></extra>"
    ),
    row=1, col=2
)



fig.add_trace(
    go.Bar(
        x=payment_rates.index,
        y=payment_rates.values,
        text=[f"{v:.2f}%" for v in payment_rates.values],
        textposition="outside",
        name="Payment method",
        showlegend=False,
        hovertemplate="%{x}<br>Churn rate: %{y:.2f}%<extra></extra>"
    ),
    row=2, col=1
)



fig.add_trace(
    go.Bar(
        x=internet_rates.index,
        y=internet_rates.values,
        text=[f"{v:.2f}%" for v in internet_rates.values],
        textposition="outside",
        name="Internet service",
        showlegend=False,
        hovertemplate="%{x}<br>Churn rate: %{y:.2f}%<extra></extra>"
    ),
    row=2, col=2
)



for status in ["No", "Yes"]:
    subset = df.loc[df["Churn"] == status, "tenure"]

    fig.add_trace(
        go.Box(
            y=subset,
            name="Stayed" if status == "No" else "Churned",
            boxpoints=False,
            showlegend=False,
            hovertemplate="Tenure: %{y} months<extra></extra>"
        ),
        row=3, col=1
    )



for status in ["No", "Yes"]:
    subset = df.loc[df["Churn"] == status, "MonthlyCharges"]

    fig.add_trace(
        go.Box(
            y=subset,
            name="Stayed" if status == "No" else "Churned",
            boxpoints=False,
            showlegend=False,
            hovertemplate="Monthly charges: %{y:.2f}<extra></extra>"
        ),
        row=3, col=2
    )



fig.add_trace(
    go.Bar(
        x=senior_rates.index,
        y=senior_rates.values,
        text=[f"{v:.2f}%" for v in senior_rates.values],
        textposition="outside",
        name="Senior status",
        showlegend=False,
        hovertemplate="%{x}<br>Churn rate: %{y:.2f}%<extra></extra>"
    ),
    row=4, col=1
)



fig.add_trace(
    go.Bar(
        x=[f"Security: {x}" for x in security_rates.index],
        y=security_rates.values,
        name="OnlineSecurity",
        showlegend=True,
        hovertemplate="%{x}<br>Churn rate: %{y:.2f}%<extra></extra>"
    ),
    row=4, col=2
)

fig.add_trace(
    go.Bar(
        x=[f"Support: {x}" for x in support_rates.index],
        y=support_rates.values,
        name="TechSupport",
        showlegend=True,
        hovertemplate="%{x}<br>Churn rate: %{y:.2f}%<extra></extra>"
    ),
    row=4, col=2
)


fig.update_layout(
    title={
        "text": "Customer Churn & Retention Intelligence Dashboard",
        "x": 0.5
    },
    height=1800,
    template="plotly_white",
    barmode="group",
    hovermode="closest",
    margin=dict(t=110, b=100, l=70, r=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.06,
        xanchor="center",
        x=0.5
    )
)

fig.update_yaxes(title_text="Customers", row=1, col=1)
fig.update_yaxes(title_text="Churn rate (%)", range=[0, 55], row=1, col=2)
fig.update_yaxes(title_text="Churn rate (%)", range=[0, 55], row=2, col=1)
fig.update_yaxes(title_text="Churn rate (%)", range=[0, 55], row=2, col=2)
fig.update_yaxes(title_text="Tenure (months)", row=3, col=1)
fig.update_yaxes(title_text="Monthly charges", row=3, col=2)
fig.update_yaxes(title_text="Churn rate (%)", range=[0, 55], row=4, col=1)
fig.update_yaxes(title_text="Churn rate (%)", range=[0, 55], row=4, col=2)

fig.update_xaxes(tickangle=-20)
fig.update_xaxes(tickangle=-30, row=4, col=2)


fig.show()

# Save a standalone interactive HTML dashboard
fig.write_html(
    "outputs/day3_churn_dashboard.html",
    include_plotlyjs=True
)

print("\nDashboard saved to:")
print("outputs/day3_churn_dashboard.html")

print("\n========== BUSINESS INSIGHTS ==========")

print(f"Overall churn rate: {overall_churn:.2f}%")

print(
    f"Highest contract churn: {contract_rates.idxmax()} "
    f"({contract_rates.max():.2f}%)"
)

print(
    f"Highest payment-method churn: {payment_rates.idxmax()} "
    f"({payment_rates.max():.2f}%)"
)

print(
    f"Highest internet-service churn: {internet_rates.idxmax()} "
    f"({internet_rates.max():.2f}%)"
)

print(
    f"Senior-citizen churn: "
    f"{senior_rates.loc['Senior']:.2f}%"
)

print("\nRemember: these patterns show associations, not causation.")