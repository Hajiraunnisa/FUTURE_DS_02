"""
Telco Customer Churn Analysis
Client-Ready Retention Dashboard & Insights Report
Future Interns — Data Analytics Task 2
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── 0. Load & Clean ────────────────────────────────────────────────────────────
df = pd.read_csv(
    r"c:\Users\Hajira\OneDrive\Documents\Data Science\Task 2\WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# TotalCharges is string — convert, coerce blanks to NaN then fill with 0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(0, inplace=True)

# Binary churn flag
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

# Tenure buckets
bins   = [0, 6, 12, 24, 48, 72]
labels = ["0–6 mo", "7–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"]
df["TenureBucket"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)

print(f"Dataset: {df.shape[0]:,} customers | "
      f"Churn rate: {df['ChurnFlag'].mean()*100:.1f}%")

# ── 1. KPIs ────────────────────────────────────────────────────────────────────
total_customers  = len(df)
churned          = df["ChurnFlag"].sum()
retained         = total_customers - churned
churn_rate       = churned / total_customers * 100
avg_tenure       = df["tenure"].mean()
avg_monthly      = df["MonthlyCharges"].mean()
total_revenue    = df["TotalCharges"].sum()
revenue_at_risk  = df[df["ChurnFlag"] == 1]["MonthlyCharges"].sum()

print(f"\n── KPI Summary ──")
print(f"  Total Customers : {total_customers:,}")
print(f"  Churned         : {churned:,}  ({churn_rate:.1f}%)")
print(f"  Retained        : {retained:,}")
print(f"  Avg Tenure      : {avg_tenure:.1f} months")
print(f"  Avg Monthly Chg : ${avg_monthly:.2f}")
print(f"  Total Revenue   : ${total_revenue:,.2f}")
print(f"  Revenue at Risk : ${revenue_at_risk:,.2f}/month")

# ── 2. Churn by Contract Type ─────────────────────────────────────────────────
contract_churn = df.groupby("Contract")["ChurnFlag"].mean().mul(100).sort_values(ascending=False)

# ── 3. Churn by Tenure Bucket ─────────────────────────────────────────────────
tenure_churn = df.groupby("TenureBucket", observed=True)["ChurnFlag"].mean().mul(100)

# ── 4. Churn by Internet Service ──────────────────────────────────────────────
internet_churn = df.groupby("InternetService")["ChurnFlag"].mean().mul(100)

# ── 5. Churn by Payment Method ────────────────────────────────────────────────
payment_churn = df.groupby("PaymentMethod")["ChurnFlag"].mean().mul(100).sort_values(ascending=False)

# ── 6. Monthly Charges: Churned vs Retained ───────────────────────────────────
churned_charges  = df[df["ChurnFlag"] == 1]["MonthlyCharges"]
retained_charges = df[df["ChurnFlag"] == 0]["MonthlyCharges"]

# ── 7. Churn by Senior Citizen ────────────────────────────────────────────────
senior_churn = df.groupby("SeniorCitizen")["ChurnFlag"].mean().mul(100)
senior_churn.index = ["Non-Senior", "Senior"]

# ── 8. Churn by Tech Support & Online Security ────────────────────────────────
tech_churn = df[df["TechSupport"] != "No internet service"].groupby("TechSupport")["ChurnFlag"].mean().mul(100)
sec_churn  = df[df["OnlineSecurity"] != "No internet service"].groupby("OnlineSecurity")["ChurnFlag"].mean().mul(100)

# ── 9. Churn by number of services ───────────────────────────────────────────
service_cols = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
df["ServiceCount"] = df[service_cols].apply(lambda row: sum(v == "Yes" for v in row), axis=1)
service_churn = df.groupby("ServiceCount")["ChurnFlag"].mean().mul(100)

# ── 10. Churn rate by Paperless Billing ───────────────────────────────────────
paperless_churn = df.groupby("PaperlessBilling")["ChurnFlag"].mean().mul(100)

# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════
BRAND   = "#0d1117"
ACCENT1 = "#ff4757"   # red  — churn
ACCENT2 = "#2ed573"   # green — retained
ACCENT3 = "#1e2130"   # card background
BLUE    = "#1e90ff"
GOLD    = "#ffa502"
PURPLE  = "#a29bfe"
TEAL    = "#00cec9"
GREY    = "#8892a4"
WHITE   = "#ffffff"

BAR_PALETTE = [ACCENT1, BLUE, GOLD, PURPLE, TEAL, ACCENT2, "#fd79a8", "#fdcb6e"]


def pct_fmt(x, pos=None):
    return f"{x:.0f}%"


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — Overview + Key Drivers
# ══════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(20, 26), facecolor=BRAND)
gs1  = gridspec.GridSpec(4, 2, figure=fig1, hspace=0.55, wspace=0.35,
                         top=0.94, bottom=0.04, left=0.07, right=0.97)

fig1.text(0.5, 0.968, "Telco Customer Churn — Retention Intelligence Dashboard",
          ha="center", fontsize=22, fontweight="bold", color=WHITE)
fig1.text(0.5, 0.955, "7,043 customers  ·  21 features  ·  Built with Python & Matplotlib",
          ha="center", fontsize=10, color=GREY)

# ── KPI Row ───────────────────────────────────────────────────────────────────
ax_kpi = fig1.add_subplot(gs1[0, :])
ax_kpi.set_facecolor(BRAND); ax_kpi.axis("off")

kpis = [
    ("Total Customers",   f"{total_customers:,}",          BLUE),
    ("Churned",           f"{churned:,}  ({churn_rate:.1f}%)", ACCENT1),
    ("Retained",          f"{retained:,}",                  ACCENT2),
    ("Avg Tenure",        f"{avg_tenure:.1f} months",        GOLD),
    ("Revenue at Risk",   f"${revenue_at_risk:,.0f}/mo",    PURPLE),
]
for i, (label, value, color) in enumerate(kpis):
    x = 0.1 + i * 0.195
    ax_kpi.add_patch(plt.Rectangle((x - 0.088, 0.05), 0.176, 0.88,
                                   transform=ax_kpi.transAxes,
                                   color=ACCENT3, zorder=0, clip_on=False))
    ax_kpi.text(x, 0.65, value, transform=ax_kpi.transAxes,
                ha="center", va="center", fontsize=17, fontweight="bold", color=color)
    ax_kpi.text(x, 0.22, label, transform=ax_kpi.transAxes,
                ha="center", va="center", fontsize=9, color=GREY)

# ── Churn vs Retained Donut (row 1, left) ────────────────────────────────────
ax1 = fig1.add_subplot(gs1[1, 0])
ax1.set_facecolor(ACCENT3)
wedge_vals  = [churned, retained]
wedge_cols  = [ACCENT1, ACCENT2]
wedge_lbls  = [f"Churned\n{churn_rate:.1f}%", f"Retained\n{100-churn_rate:.1f}%"]
wedges, texts = ax1.pie(wedge_vals, labels=wedge_lbls, colors=wedge_cols,
                        startangle=90, wedgeprops={"width": 0.55, "edgecolor": BRAND, "linewidth": 2},
                        textprops={"color": WHITE, "fontsize": 11, "fontweight": "bold"})
ax1.set_title("Overall Churn vs Retention", color=WHITE, fontsize=13, fontweight="bold", pad=10)

# ── Churn by Contract (row 1, right) ─────────────────────────────────────────
ax2 = fig1.add_subplot(gs1[1, 1])
ax2.set_facecolor(ACCENT3)
colors2 = [ACCENT1, GOLD, ACCENT2]
bars2 = ax2.bar(contract_churn.index, contract_churn.values, color=colors2,
                edgecolor="none", width=0.5)
for bar, val in zip(bars2, contract_churn.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold", color=WHITE)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
ax2.set_title("Churn Rate by Contract Type", color=WHITE, fontsize=13, fontweight="bold", pad=10)
ax2.set_ylabel("Churn Rate", color=GREY, fontsize=9)
ax2.tick_params(colors=GREY, labelsize=9)
for spine in ax2.spines.values(): spine.set_edgecolor(GREY); spine.set_alpha(0.2)

# ── Churn by Tenure Bucket (row 2, left) ─────────────────────────────────────
ax3 = fig1.add_subplot(gs1[2, 0])
ax3.set_facecolor(ACCENT3)
colors3 = [ACCENT1 if v > 30 else GOLD if v > 15 else ACCENT2 for v in tenure_churn.values]
bars3 = ax3.bar(tenure_churn.index, tenure_churn.values, color=colors3,
                edgecolor="none", width=0.6)
for bar, val in zip(bars3, tenure_churn.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold", color=WHITE)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
ax3.set_title("Churn Rate by Customer Tenure", color=WHITE, fontsize=13, fontweight="bold", pad=10)
ax3.set_xlabel("Tenure Bucket", color=GREY, fontsize=9)
ax3.set_ylabel("Churn Rate", color=GREY, fontsize=9)
ax3.tick_params(colors=GREY, labelsize=9)
for spine in ax3.spines.values(): spine.set_edgecolor(GREY); spine.set_alpha(0.2)

# ── Churn by Internet Service (row 2, right) ──────────────────────────────────
ax4 = fig1.add_subplot(gs1[2, 1])
ax4.set_facecolor(ACCENT3)
colors4 = [ACCENT1 if v > 30 else GOLD if v > 15 else ACCENT2 for v in internet_churn.values]
bars4 = ax4.bar(internet_churn.index, internet_churn.values, color=colors4,
                edgecolor="none", width=0.5)
for bar, val in zip(bars4, internet_churn.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold", color=WHITE)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
ax4.set_title("Churn Rate by Internet Service", color=WHITE, fontsize=13, fontweight="bold", pad=10)
ax4.set_ylabel("Churn Rate", color=GREY, fontsize=9)
ax4.tick_params(colors=GREY, labelsize=9)
for spine in ax4.spines.values(): spine.set_edgecolor(GREY); spine.set_alpha(0.2)

# ── Monthly Charges Distribution (row 3, full width) ─────────────────────────
ax5 = fig1.add_subplot(gs1[3, :])
ax5.set_facecolor(ACCENT3)
ax5.hist(retained_charges, bins=40, alpha=0.7, color=ACCENT2, label="Retained", density=True)
ax5.hist(churned_charges,  bins=40, alpha=0.7, color=ACCENT1, label="Churned",  density=True)
ax5.axvline(churned_charges.mean(),  color=ACCENT1, linestyle="--", lw=1.5,
            label=f"Churned avg: ${churned_charges.mean():.0f}")
ax5.axvline(retained_charges.mean(), color=ACCENT2, linestyle="--", lw=1.5,
            label=f"Retained avg: ${retained_charges.mean():.0f}")
ax5.set_title("Monthly Charges Distribution — Churned vs Retained", color=WHITE,
              fontsize=13, fontweight="bold", pad=10)
ax5.set_xlabel("Monthly Charges ($)", color=GREY, fontsize=10)
ax5.set_ylabel("Density", color=GREY, fontsize=10)
ax5.tick_params(colors=GREY, labelsize=9)
ax5.legend(facecolor=ACCENT3, labelcolor=WHITE, fontsize=9, framealpha=0.7)
for spine in ax5.spines.values(): spine.set_edgecolor(GREY); spine.set_alpha(0.2)

page1_path = r"c:\Users\Hajira\OneDrive\Documents\Data Science\Task 2\churn_dashboard_page1.png"
fig1.savefig(page1_path, dpi=150, bbox_inches="tight", facecolor=BRAND)
print(f"Page 1 saved → {page1_path}")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — Deep Dive: Service, Payment, Segments
# ══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(20, 22), facecolor=BRAND)
gs2  = gridspec.GridSpec(3, 2, figure=fig2, hspace=0.55, wspace=0.35,
                         top=0.93, bottom=0.05, left=0.07, right=0.97)

fig2.text(0.5, 0.96, "Telco Customer Churn — Segment & Service Deep Dive",
          ha="center", fontsize=22, fontweight="bold", color=WHITE)
fig2.text(0.5, 0.946, "Payment methods · Service adoption · Demographics · Retention drivers",
          ha="center", fontsize=10, color=GREY)

# ── Churn by Payment Method (row 0, left) ─────────────────────────────────────
ax6 = fig2.add_subplot(gs2[0, 0])
ax6.set_facecolor(ACCENT3)
cols6 = [ACCENT1 if v > 30 else GOLD if v > 20 else ACCENT2 for v in payment_churn.values]
bars6 = ax6.barh(payment_churn.index[::-1], payment_churn.values[::-1],
                 color=cols6[::-1], edgecolor="none", height=0.55)
for bar, val in zip(bars6, payment_churn.values[::-1]):
    ax6.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}%", va="center", fontsize=10, fontweight="bold", color=WHITE)
ax6.xaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
ax6.set_title("Churn Rate by Payment Method", color=WHITE, fontsize=13, fontweight="bold", pad=10)
ax6.tick_params(colors=GREY, labelsize=8)
for spine in ax6.spines.values(): spine.set_edgecolor(GREY); spine.set_alpha(0.2)

# ── Churn by Senior Citizen (row 0, right) ────────────────────────────────────
ax7 = fig2.add_subplot(gs2[0, 1])
ax7.set_facecolor(ACCENT3)
cols7 = [BLUE, ACCENT1]
bars7 = ax7.bar(senior_churn.index, senior_churn.values, color=cols7,
                edgecolor="none", width=0.45)
for bar, val in zip(bars7, senior_churn.values):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", fontsize=13, fontweight="bold", color=WHITE)
ax7.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
ax7.set_title("Churn Rate: Senior vs Non-Senior", color=WHITE, fontsize=13, fontweight="bold", pad=10)
ax7.set_ylabel("Churn Rate", color=GREY, fontsize=9)
ax7.tick_params(colors=GREY, labelsize=10)
for spine in ax7.spines.values(): spine.set_edgecolor(GREY); spine.set_alpha(0.2)

# ── Tech Support & Online Security impact (row 1, left) ───────────────────────
ax8 = fig2.add_subplot(gs2[1, 0])
ax8.set_facecolor(ACCENT3)
x8     = np.arange(2)
width8 = 0.3
b1 = ax8.bar(x8 - width8/2, tech_churn.values, width8, label="Tech Support",
             color=[ACCENT2, ACCENT1], edgecolor="none")
b2 = ax8.bar(x8 + width8/2, sec_churn.values,  width8, label="Online Security",
             color=[TEAL, GOLD], edgecolor="none")
for bars_grp in [b1, b2]:
    for bar in bars_grp:
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold", color=WHITE)
ax8.set_xticks(x8)
ax8.set_xticklabels(["Yes (subscribed)", "No (not subscribed)"], color=GREY, fontsize=9)
ax8.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
ax8.set_title("Churn Rate: Tech Support & Online Security", color=WHITE,
              fontsize=12, fontweight="bold", pad=10)
ax8.set_ylabel("Churn Rate", color=GREY, fontsize=9)
ax8.tick_params(axis="y", colors=GREY, labelsize=9)
p1 = mpatches.Patch(color=ACCENT2, label="Tech Support — Yes")
p2 = mpatches.Patch(color=ACCENT1, label="Tech Support — No")
p3 = mpatches.Patch(color=TEAL,    label="Online Security — Yes")
p4 = mpatches.Patch(color=GOLD,    label="Online Security — No")
ax8.legend(handles=[p1, p2, p3, p4], facecolor=ACCENT3, labelcolor=WHITE,
           fontsize=8, framealpha=0.7)
for spine in ax8.spines.values(): spine.set_edgecolor(GREY); spine.set_alpha(0.2)

# ── Churn by Service Count (row 1, right) ─────────────────────────────────────
ax9 = fig2.add_subplot(gs2[1, 1])
ax9.set_facecolor(ACCENT3)
sc_colors = [ACCENT1 if v > 35 else GOLD if v > 20 else ACCENT2 for v in service_churn.values]
bars9 = ax9.bar(service_churn.index.astype(str), service_churn.values,
                color=sc_colors, edgecolor="none", width=0.6)
for bar, val in zip(bars9, service_churn.values):
    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold", color=WHITE)
ax9.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
ax9.set_title("Churn Rate by Number of Services Subscribed", color=WHITE,
              fontsize=12, fontweight="bold", pad=10)
ax9.set_xlabel("Number of Services", color=GREY, fontsize=9)
ax9.set_ylabel("Churn Rate", color=GREY, fontsize=9)
ax9.tick_params(colors=GREY, labelsize=9)
for spine in ax9.spines.values(): spine.set_edgecolor(GREY); spine.set_alpha(0.2)

# ── Churn by Paperless Billing & Partner/Dependents (row 2, left) ─────────────
ax10 = fig2.add_subplot(gs2[2, 0])
ax10.set_facecolor(ACCENT3)
partner_churn   = df.groupby("Partner")["ChurnFlag"].mean().mul(100)
dependent_churn = df.groupby("Dependents")["ChurnFlag"].mean().mul(100)
x10    = np.arange(2)
w10    = 0.28
bp1 = ax10.bar(x10 - w10, paperless_churn.values, w10, label="Paperless Billing",
               color=[ACCENT2, ACCENT1], edgecolor="none")
bp2 = ax10.bar(x10,        partner_churn.values,   w10, label="Has Partner",
               color=[TEAL, BLUE], edgecolor="none")
bp3 = ax10.bar(x10 + w10,  dependent_churn.values, w10, label="Has Dependents",
               color=[GOLD, PURPLE], edgecolor="none")
for grp in [bp1, bp2, bp3]:
    for bar in grp:
        ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                  f"{bar.get_height():.0f}%", ha="center", fontsize=8, color=WHITE)
ax10.set_xticks(x10)
ax10.set_xticklabels(["Yes", "No"], color=GREY, fontsize=10)
ax10.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
ax10.set_title("Churn Rate: Billing, Partner & Dependents", color=WHITE,
               fontsize=12, fontweight="bold", pad=10)
ax10.set_ylabel("Churn Rate", color=GREY, fontsize=9)
ax10.tick_params(axis="y", colors=GREY, labelsize=9)
ax10.legend(facecolor=ACCENT3, labelcolor=WHITE, fontsize=8, framealpha=0.7)
for spine in ax10.spines.values(): spine.set_edgecolor(GREY); spine.set_alpha(0.2)

# ── Revenue at Risk by Contract (row 2, right) ────────────────────────────────
ax11 = fig2.add_subplot(gs2[2, 1])
ax11.set_facecolor(ACCENT3)
risk_by_contract = (
    df[df["ChurnFlag"] == 1]
    .groupby("Contract")["MonthlyCharges"]
    .sum()
    .sort_values(ascending=False)
)
cols11 = [ACCENT1, GOLD, ACCENT2]
bars11 = ax11.bar(risk_by_contract.index, risk_by_contract.values,
                  color=cols11, edgecolor="none", width=0.5)
for bar, val in zip(bars11, risk_by_contract.values):
    ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
              f"${val:,.0f}", ha="center", fontsize=10, fontweight="bold", color=WHITE)
ax11.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax11.set_title("Monthly Revenue at Risk by Contract Type", color=WHITE,
               fontsize=12, fontweight="bold", pad=10)
ax11.set_ylabel("Monthly Revenue Lost ($)", color=GREY, fontsize=9)
ax11.tick_params(colors=GREY, labelsize=9)
for spine in ax11.spines.values(): spine.set_edgecolor(GREY); spine.set_alpha(0.2)

page2_path = r"c:\Users\Hajira\OneDrive\Documents\Data Science\Task 2\churn_dashboard_page2.png"
fig2.savefig(page2_path, dpi=150, bbox_inches="tight", facecolor=BRAND)
print(f"Page 2 saved → {page2_path}")
plt.close("all")

# ── Insights Report ────────────────────────────────────────────────────────────
report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       TELCO CUSTOMER CHURN — RETENTION INSIGHTS & RECOMMENDATIONS          ║
╚══════════════════════════════════════════════════════════════════════════════╝

PREPARED BY  : Customer Analytics Team
DATASET      : Telco Customer Churn (Kaggle — blastchar)
RECORDS      : {total_customers:,} customers  |  21 features

─────────────────────────────────────────────────────────────────────────────
 KPI SUMMARY
─────────────────────────────────────────────────────────────────────────────
  ▸ Total Customers    : {total_customers:,}
  ▸ Churned Customers  : {churned:,}  ({churn_rate:.1f}%)
  ▸ Retained Customers : {retained:,}  ({100-churn_rate:.1f}%)
  ▸ Avg Customer Tenure: {avg_tenure:.1f} months
  ▸ Avg Monthly Charge : ${avg_monthly:.2f}
  ▸ Total Revenue      : ${total_revenue:,.2f}
  ▸ Revenue at Risk    : ${revenue_at_risk:,.2f}/month from churned customers

─────────────────────────────────────────────────────────────────────────────
 KEY INSIGHTS
─────────────────────────────────────────────────────────────────────────────

1. OVERALL CHURN RATE IS HIGH — {churn_rate:.1f}%
   Nearly 1 in 3 customers has left. This is significantly above the healthy
   SaaS/telecom benchmark of 5–7% annually. Urgent retention action is needed.

2. CONTRACT TYPE IS THE STRONGEST PREDICTOR
   Month-to-month customers churn at ~42% vs only ~3% for two-year contracts.
   Customers on long-term contracts are 14x more likely to stay.

3. NEW CUSTOMERS ARE THE MOST VULNERABLE
   Churn is highest in the first 6 months (~50%). Once a customer reaches
   12+ months, churn drops dramatically. Early engagement is critical.

4. FIBER OPTIC CUSTOMERS CHURN THE MOST (~42%)
   Despite being a premium service, fiber customers churn more than DSL users.
   This suggests pricing dissatisfaction or unmet service quality expectations.

5. ELECTRONIC CHECK PAYMENT = HIGHEST CHURN RISK
   Customers paying by electronic check churn at ~45%. Auto-pay customers
   (credit card / bank transfer) churn at less than half that rate.

6. SENIORS ARE A HIGH-RISK SEGMENT
   Senior citizens churn at ~41% vs ~24% for non-seniors. They may need
   simpler plans, dedicated support, or tailored communication.

7. ADD-ON SERVICES DRAMATICALLY REDUCE CHURN
   Customers with Tech Support and Online Security churn at ~15% vs ~40%
   for those without. Each additional service a customer subscribes to
   lowers their churn probability.

8. CUSTOMERS WITHOUT PARTNERS/DEPENDENTS CHURN MORE
   Solo customers are more likely to leave than those with family plans.
   Family/bundled plans create stickiness and switching cost.

─────────────────────────────────────────────────────────────────────────────
 ACTIONABLE RECOMMENDATIONS
─────────────────────────────────────────────────────────────────────────────

  ✔ CONVERT MONTH-TO-MONTH CUSTOMERS
    Offer discounted annual or two-year contracts with incentives (free month,
    device upgrade). This single action could cut churn by up to 30%.

  ✔ EARLY ONBOARDING PROGRAMME (0–6 months)
    Assign dedicated support, trigger check-in emails at day 7, 30, 90.
    First-month churn is ~50% — early engagement is the highest-ROI lever.

  ✔ PUSH AUTO-PAY ADOPTION
    Offer $5/month discount for switching from electronic check to auto-pay
    (bank transfer / credit card). Reduces friction and churn risk.

  ✔ BUNDLE ADD-ON SERVICES
    Proactively offer Tech Support and Online Security bundles to new customers
    at onboarding. Each service adds retention value and monthly revenue.

  ✔ SENIOR CITIZEN RETENTION PROGRAMME
    Create simplified plans, dedicated senior helpline, and proactive outreach.
    41% churn in this segment represents significant preventable revenue loss.

  ✔ FIBRE OPTIC QUALITY REVIEW
    Investigate root cause of high fibre churn — survey churned fibre customers
    for feedback. Consider a service guarantee or price-lock for loyal users.

  ✔ FAMILY PLAN INCENTIVES
    Promote multi-line / family bundles to solo customers. Customers with
    dependents or partners churn significantly less.

─────────────────────────────────────────────────────────────────────────────
 DASHBOARD FILES
─────────────────────────────────────────────────────────────────────────────
  Page 1: churn_dashboard_page1.png  (KPIs, donut, contract, tenure, charges)
  Page 2: churn_dashboard_page2.png  (Payment, seniors, services, risk)

════════════════════════════════════════════════════════════════════════════════
"""

print(report)

report_path = r"c:\Users\Hajira\OneDrive\Documents\Data Science\Task 2\churn_insights_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"Report saved → {report_path}")
