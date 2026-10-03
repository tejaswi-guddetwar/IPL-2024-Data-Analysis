# ==========================================
# IPL DATA ANALYSIS 2024
# Python | Pandas | NumPy | Matplotlib | Seaborn
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATASET
df = pd.read_csv("IPL_Data_Analysis_2024.csv")

print("IPL DATA LOADED SUCCESSFULLY")
print(df.head())

# 2. DATA EXPLORATION
print("\nFIRST 5 ROWS")
print(df.head())

print("\nLAST 5 ROWS")
print(df.tail())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nSTATISTICAL SUMMARY")
print(df.describe(include="all"))

# 3. DATA CLEANING
df = df.drop_duplicates().copy()

df["Match_Date"] = pd.to_datetime(
    df["Match_Date"], errors="coerce"
)

# Fill missing numeric values
numeric_cols = df.select_dtypes(
    include=np.number
).columns

df[numeric_cols] = df[numeric_cols].fillna(0)

# 4. CREATE NEW COLUMNS
df["Total_Runs"] = (
    df["Team1_Runs"] + df["Team2_Runs"]
)

df["Total_Wickets"] = (
    df["Team1_Wickets"] + df["Team2_Wickets"]
)

df["Month"] = df["Match_Date"].dt.month_name()

# 5. KEY PERFORMANCE INDICATORS
total_matches = df["Match_ID"].nunique()

total_teams = pd.unique(
    df[["Team1", "Team2"]].values.ravel()
).size

total_runs = df["Total_Runs"].sum()

average_runs = df["Total_Runs"].mean()

total_wickets = df["Total_Wickets"].sum()

print("\nKEY PERFORMANCE INDICATORS")
print("Total Matches:", total_matches)
print("Total Teams:", total_teams)
print("Total Runs:", total_runs)
print("Average Runs:", round(average_runs, 2))
print("Total Wickets:", total_wickets)

# 6. TEAM ANALYSIS
team_wins = df["Winner"].value_counts()

matches_played = pd.concat(
    [df["Team1"], df["Team2"]]
).value_counts()

print("\nTEAM WINS")
print(team_wins)

print("\nMATCHES PLAYED BY TEAM")
print(matches_played)

team_summary = pd.DataFrame({
    "Matches_Played": matches_played,
    "Wins": team_wins
}).fillna(0)

team_summary["Losses"] = (
    team_summary["Matches_Played"] -
    team_summary["Wins"]
)

team_summary["Win_Percentage"] = (
    team_summary["Wins"] /
    team_summary["Matches_Played"] * 100
).round(2)

print("\nTEAM PERFORMANCE")
print(team_summary)

# 7. PLAYER ANALYSIS
player_awards = df["Player_of_Match"].value_counts()

top_run_scorers = df["Top_Run_Scorer"].value_counts()

top_wicket_takers = df["Top_Wicket_Taker"].value_counts()

print("\nPLAYER OF THE MATCH AWARDS")
print(player_awards)

print("\nTOP RUN SCORER APPEARANCES")
print(top_run_scorers.head(10))

print("\nTOP WICKET TAKER APPEARANCES")
print(top_wicket_takers.head(10))

# Note: These are appearance counts, not career
# or season totals of runs and wickets.

# 8. TOSS ANALYSIS
toss_decisions = df["Toss_Decision"].value_counts()

toss_winner_match_wins = (
    df["Toss_Match_Win"].sum()
)

toss_win_percentage = (
    toss_winner_match_wins / total_matches * 100
    if total_matches > 0 else 0
)

print("\nTOSS DECISIONS")
print(toss_decisions)

print(
    "\nToss winner also won:",
    toss_winner_match_wins
)

print(
    "Toss-to-match win percentage:",
    round(toss_win_percentage, 2), "%"
)

# 9. VENUE / TEAM RESULT ANALYSIS
print("\nWINNING TEAM COUNTS")
print(df["Winner"].value_counts())

print("\nRESULT TYPES")
print(df["Result"].value_counts())

# 10. MATCH RUN ANALYSIS
highest_total = df["Total_Runs"].max()
lowest_total = df["Total_Runs"].min()

highest_run_match = df.loc[
    df["Total_Runs"].idxmax()
]

print("\nHIGHEST COMBINED MATCH RUNS")
print(highest_total)
print(highest_run_match)

print("\nLOWEST COMBINED MATCH RUNS")
print(lowest_total)

# 11. CHART SETTINGS
sns.set_theme(style="whitegrid")

# Chart 1: Matches Won by Team
plt.figure(figsize=(10, 6))
team_wins.sort_values().plot(kind="barh")
plt.title("Matches Won by Team")
plt.xlabel("Number of Wins")
plt.ylabel("Team")
plt.tight_layout()
plt.savefig("01_team_wins.png")
plt.show()

# Chart 2: Total Runs by Match
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Match_ID", y="Total_Runs")
plt.title("Total Runs by Match")
plt.xlabel("Match ID")
plt.ylabel("Total Runs")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("02_total_runs.png")
plt.show()

# Chart 3: Player of the Match Awards
plt.figure(figsize=(10, 6))
player_awards.head(10).sort_values().plot(
    kind="barh"
)
plt.title("Top 10 Player of the Match Awards")
plt.xlabel("Awards")
plt.ylabel("Player")
plt.tight_layout()
plt.savefig("03_player_awards.png")
plt.show()

# Chart 4: Toss Decision Distribution
plt.figure(figsize=(7, 6))
toss_decisions.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Toss Decision Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("04_toss_decisions.png")
plt.show()

# Chart 5: Top Run Scorer Appearances
plt.figure(figsize=(10, 6))
top_run_scorers.head(10).sort_values().plot(
    kind="barh"
)
plt.title("Top Run Scorer Appearances")
plt.xlabel("Number of Matches")
plt.ylabel("Player")
plt.tight_layout()
plt.savefig("05_top_scorer_appearances.png")
plt.show()

# Chart 6: Top Wicket Taker Appearances
plt.figure(figsize=(10, 6))
top_wicket_takers.head(10).sort_values().plot(
    kind="barh"
)
plt.title("Top Wicket Taker Appearances")
plt.xlabel("Number of Matches")
plt.ylabel("Player")
plt.tight_layout()
plt.savefig("06_top_wicket_appearances.png")
plt.show()

# Chart 7: Runs Distribution
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x="Total_Runs",
    bins=10,
    kde=True
)
plt.title("Distribution of Total Match Runs")
plt.xlabel("Total Runs")
plt.ylabel("Number of Matches")
plt.tight_layout()
plt.savefig("07_runs_distribution.png")
plt.show()

# Chart 8: Runs vs Wickets
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="Total_Runs",
    y="Total_Wickets",
    hue="Winner"
)
plt.title("Runs vs Wickets")
plt.xlabel("Total Runs")
plt.ylabel("Total Wickets")
plt.legend(
    title="Winner",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)
plt.tight_layout()
plt.savefig("08_runs_vs_wickets.png")
plt.show()

# Chart 9: Team Performance
plt.figure(figsize=(10, 6))
team_summary["Wins"].sort_values().plot(
    kind="barh"
)
plt.title("Team Performance Comparison")
plt.xlabel("Wins")
plt.ylabel("Team")
plt.tight_layout()
plt.savefig("09_team_performance.png")
plt.show()

# Chart 10: Winning Margin
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x="Run_Difference",
    bins=10,
    kde=True
)
plt.title("Distribution of Run Differences")
plt.xlabel("Run Difference")
plt.ylabel("Number of Matches")
plt.tight_layout()
plt.savefig("10_run_difference.png")
plt.show()

# 12. CORRELATION HEATMAP
numeric_data = df.select_dtypes(
    include=np.number
)

correlation = numeric_data.corr()

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("IPL Data Correlation Heatmap")
plt.tight_layout()
plt.savefig("11_correlation_heatmap.png")
plt.show()

# 13. EXPORT CLEANED DATA
df.to_csv(
    "IPL_Cleaned_Data.csv",
    index=False
)

team_summary.to_csv(
    "IPL_Team_Summary.csv"
)

# 14. FINAL RESULTS
print("\n========== PROJECT COMPLETED ==========")
print("Total Matches:", total_matches)
print("Total Teams:", total_teams)
print("Total Runs:", total_runs)
print("Total Wickets:", total_wickets)
print("Average Runs:", round(average_runs, 2))
print("Toss Win Percentage:", round(
    toss_win_percentage, 2
), "%")
print("Cleaned CSV files exported successfully")
print("All charts saved successfully")
