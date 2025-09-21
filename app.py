# app.py
import streamlit as st
import pandas as pd
from datetime import datetime

# ------------------ YOUR LINKS ------------------
CSV_TEAMS       = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSoMq0Hiw6Q0XdTbxFC12Ya8Cg66v4MtG7xI_Ir-xelXK4ZootrkwsGrPmXXz5uLIb-DC_bcakPu2ub/pub?gid=0&single=true&output=csv"
CSV_WEEKS       = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSoMq0Hiw6Q0XdTbxFC12Ya8Cg66v4MtG7xI_Ir-xelXK4ZootrkwsGrPmXXz5uLIb-DC_bcakPu2ub/pub?gid=29563283&single=true&output=csv"
CSV_CHALLENGES  = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSoMq0Hiw6Q0XdTbxFC12Ya8Cg66v4MtG7xI_Ir-xelXK4ZootrkwsGrPmXXz5uLIb-DC_bcakPu2ub/pub?gid=570391343&single=true&output=csv"
CSV_TOLLS       = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSoMq0Hiw6Q0XdTbxFC12Ya8Cg66v4MtG7xI_Ir-xelXK4ZootrkwsGrPmXXz5uLIb-DC_bcakPu2ub/pub?gid=332831073&single=true&output=csv"

LEAGUE_TITLE    = "🏈 Show Me Your TDs 2025 League Portal"
LEAGUE_FEE      = 100.0  # per-team buy-in (used in Net; not shown as a column)
TOLL_BASE       = 5.0    # first Desai Default offense = $5, then doubles per offense

# External resources (shown as nice text links)
URL_YAHOO       = "https://football.fantasysports.yahoo.com/f1/528158"
URL_RULES       = "https://drive.google.com/file/d/11_X0igA7hw4DQbVFek-l6VTtf0tGdDPQ/view?usp=drive_link"
# ------------------------------------------------

st.set_page_config(page_title="League Portal", layout="wide", initial_sidebar_state="collapsed")

# ------------------ helpers ------------------
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def parse_date(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

def ensure_week_headers(weeks: pd.DataFrame) -> pd.DataFrame:
    alt = {"start": "start_date", "startdate": "start_date", "end": "end_date", "enddate": "end_date"}
    for src, dst in alt.items():
        if src in weeks.columns and dst not in weeks.columns:
            weeks[dst] = weeks[src]
    return weeks

def norm_id_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip().str.lower()
    # treat blanks/none/nan as missing
    return out.mask(out.isin(["", "nan", "none"]))

def to_bool_loose(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["y", "yes", "true", "1"])

def current_week_from_calendar(weeks: pd.DataFrame) -> int | None:
    if weeks.empty or not {"week", "start_date", "end_date"}.issubset(weeks.columns):
        return None
    today_ts = pd.Timestamp(datetime.now().date())
    hit = weeks[(weeks["start_date"] <= today_ts) & (today_ts <= weeks["end_date"])]
    if not hit.empty and pd.notna(hit.iloc[0]["week"]):
        return int(hit.iloc[0]["week"])
    past = weeks[(weeks["start_date"] <= today_ts) & weeks["week"].notna()].sort_values("start_date")
    if not past.empty:
        return int(past.iloc[-1]["week"])
    w = weeks["week"].dropna()
    return int(w.min()) if not w.empty else None

def money(x):
    try:
        val = float(x)
        if pd.isna(val):
            val = 0.0
        return f"${val:,.0f}"
    except Exception:
        return x

def table_height(n_rows: int, row_px: int = 34, header_px: int = 38, pad_px: int = 8, max_px: int = 900) -> int:
    return min(max_px, header_px + max(0, n_rows) * row_px + pad_px)

def friendly_date(dt: pd.Timestamp | datetime | None) -> str:
    if pd.isna(dt) or dt is None:
        return ""
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    return f"{dt.strftime('%A')}, {dt.strftime('%B')} {dt.day}"

# Desai Default Tolls: base* (2^(N+M) - 2^N)
def compute_toll_row(total_prior_offenses: float, misses_this_week: float, base: float) -> float:
    if pd.isna(misses_this_week) or misses_this_week <= 0:
        return 0.0
    N = float(total_prior_offenses or 0)
    M = float(misses_this_week or 0)
    return base * (2.0 ** (N + M) - 2.0 ** N)

# --- NEW: canonical ID helpers (support initials OR legacy team_id) ---
def coalesce_col(df: pd.DataFrame, targets: list[str], new_col: str) -> None:
    """
    Create df[new_col] from the first existing column in 'targets', normalized.
    If none exist, new_col will not be created.
    """
    for c in targets:
        if c in df.columns:
            df[new_col] = norm_id_series(df[c])
            return

# ------------------ data load ------------------
@st.cache_data(ttl=300)
def load():
    teams = norm_cols(pd.read_csv(CSV_TEAMS))
    # Map Teams.team_owner -> owner (so downstream joins work)
    if "team_owner" in teams.columns and "owner" not in teams.columns:
        teams["owner"] = teams["team_owner"]

    weeks = norm_cols(pd.read_csv(CSV_WEEKS))
    chal  = norm_cols(pd.read_csv(CSV_CHALLENGES))

    tolls = pd.DataFrame()
    if CSV_TOLLS:
        try:
            tolls = norm_cols(pd.read_csv(CSV_TOLLS))
        except Exception:
            tolls = pd.DataFrame()

    # normalize headers / types
    if "prize amount" in chal.columns and "prize_amount" not in chal.columns:
        chal.rename(columns={"prize amount": "prize_amount"}, inplace=True)

    weeks = ensure_week_headers(weeks)
    parse_date(weeks, "start_date"); parse_date(weeks, "end_date")
    if "week" in weeks: weeks["week"] = pd.to_numeric(weeks["week"], errors="coerce").astype("Int64")
    if "week" in chal:  chal["week"]  = pd.to_numeric(chal["week"],  errors="coerce").astype("Int64")

    # --- IDs: build canonical columns ---
    # TEAMS: tid from 'initials' or 'team_id' (backward compatible)
    coalesce_col(teams, ["initials", "team_id"], "tid")
    # CHALLENGES: winner_tid from 'winner_initials' or legacy 'winner_team_id'
    coalesce_col(chal, ["winner_initials", "winner_team_id"], "winner_tid")
    # TOLLS: tid from 'initials' or 'team_id'
    if not tolls.empty:
        coalesce_col(tolls, ["initials", "team_id"], "tid")

    # survivor/elimination fields
    if "survivor_eliminated_week" in teams.columns and "eliminated_week" not in teams.columns:
        teams["eliminated_week"] = pd.to_numeric(teams["survivor_eliminated_week"], errors="coerce").astype("Int64")
    elif "eliminated_week" in teams.columns:
        teams["eliminated_week"] = pd.to_numeric(teams["eliminated_week"], errors="coerce").astype("Int64")

    if "prize_amount" in chal: chal["prize_amount"] = pd.to_numeric(chal["prize_amount"], errors="coerce").fillna(0.0)
    if "paid" in chal:         chal["paid"] = to_bool_loose(chal["paid"])
    else:                      chal["paid"] = False

    if not tolls.empty:
        if "missing_starters" not in tolls.columns:
            for alt in ["players_missing", "incomplete_starters", "missing"]:
                if alt in tolls.columns:
                    tolls.rename(columns={alt: "missing_starters"}, inplace=True)
                    break
        if "week" in tolls: tolls["week"] = pd.to_numeric(tolls["week"], errors="coerce").astype("Int64")
        if "missing_starters" in tolls:
            tolls["missing_starters"] = pd.to_numeric(tolls["missing_starters"], errors="coerce").fillna(0).astype(int)
        else:
            tolls["missing_starters"] = 0

    return teams, weeks, chal, tolls

try:
    teams, weeks, chal, tolls = load()
except Exception as e:
    st.error("Failed to load data. Check your CSV links and that each tab is published as CSV.")
    st.exception(e)
    st.stop()

wk_current = current_week_from_calendar(weeks)
if wk_current is None:
    wk_current = int(chal["week"].dropna().max()) if "week" in chal.columns and not chal.empty else 1

# ------------------ header ------------------
st.title(LEAGUE_TITLE)

# Friendly subheader for current week
sd = ed = None
if not weeks.empty and {"week","start_date","end_date"}.issubset(weeks.columns):
    row = weeks.loc[weeks["week"] == wk_current]
    if not row.empty:
        sd, ed = row.iloc[0]["start_date"], row.iloc[0]["end_date"]

if pd.notna(sd) and pd.notna(ed):
    st.markdown(f"#### Week {wk_current} — {friendly_date(sd)} to {friendly_date(ed)}")
else:
    st.markdown(f"#### Week {wk_current}")

# 📚 Resources (buttons)
st.link_button("Open Yahoo League", URL_YAHOO)
st.link_button("Open League Rules (PDF)", URL_RULES)

st.markdown("---")

# Current-week anchor & picker range
MAX_WEEK = int(wk_current or 1)
WEEK_MAX_AVAILABLE = int(max([
    (weeks["week"].dropna().max() if "week" in weeks else 0) or 0,
    (chal["week"].dropna().max() if "week" in chal else 0) or 0,
    MAX_WEEK
]))

# ------------------ this week ------------------
st.subheader("📌 This Week’s Challenges")
wk = st.number_input("Week", min_value=1, max_value=WEEK_MAX_AVAILABLE,
                     value=min(int(wk_current or 1), WEEK_MAX_AVAILABLE), step=1)
wk_chal = chal[chal["week"] == wk].copy() if "week" in chal.columns else chal.copy()

# join winner names/owners via canonical IDs
if not wk_chal.empty and "winner_tid" in wk_chal.columns and "tid" in teams.columns:
    try:
        wk_chal = wk_chal.merge(teams[["tid","team_name","owner"]],
                                left_on="winner_tid", right_on="tid", how="left")
    except Exception:
        wk_chal["team_name"] = None
        wk_chal["owner"] = None

if wk_chal.empty:
    st.info("No challenges found for this week yet.")
else:
    if "challenge_id" in wk_chal.columns:
        wk_chal["has_winner"] = wk_chal["winner_tid"].notna()
        wk_chal = (wk_chal.sort_values(["has_winner","paid","challenge_id"], ascending=[False, False, True])
                           .drop_duplicates(subset=["challenge_id"], keep="first"))
    for _, r in wk_chal.sort_values("challenge_name").iterrows():
        with st.container():
            left, mid, right = st.columns([3,1,2])
            with left:
                st.markdown(f"**{r.get('challenge_name','Challenge')}**")
                desc = r.get("description")
                if isinstance(desc, str) and desc.strip():
                    st.caption(desc)
            with mid:
                st.metric("Prize", money(r.get("prize_amount", 0)))
            with right:
                winner = r.get("team_name"); details = r.get("winner_details", "")
                owner  = r.get("owner")
                if isinstance(winner, str) and winner.strip():
                    st.write(f"🏆 **{winner}**")
                    if isinstance(owner, str) and owner.strip():
                        st.caption(f"Owner: {owner}")
                    if isinstance(details, str) and details.strip():
                        st.caption(details)
                if bool(r.get("paid", False)):
                    st.success("Paid")

st.markdown("---")

# ------------------ history ------------------
st.subheader("📜 Challenge Winners History")
hist = chal.copy()
if "week" in hist.columns:
    hist = hist[hist["week"].notna()]
    hist = hist[hist["week"] <= MAX_WEEK]
if "winner_tid" in hist.columns:
    hist = hist.dropna(subset=["winner_tid"])
if "winner_tid" in hist.columns and "tid" in teams.columns:
    try:
        hist = hist.merge(teams[["tid","team_name","owner"]],
                          left_on="winner_tid", right_on="tid", how="left")
    except Exception:
        hist["team_name"] = None
        hist["owner"] = None

cols = [c for c in ["week","challenge_name","team_name","owner","winner_details","prize_amount","paid"] if c in hist.columns]
hist_show = hist[cols].rename(columns={
    "week":"Week",
    "challenge_name":"Challenge",
    "team_name":"Winner",
    "owner":"Owner",
    "winner_details":"Details",
    "prize_amount":"Prize",
    "paid":"Paid"
})
# High Score first within each week
hist_show["__prio"] = (hist_show["Challenge"].astype(str).str.strip().str.lower() != "high score").astype(int)
hist_show = hist_show.sort_values(["Week", "__prio", "Challenge"]).drop(columns="__prio")
if "Prize" in hist_show.columns:
    hist_show["Prize"] = hist_show["Prize"].apply(money)

st.dataframe(hist_show, use_container_width=True, hide_index=True, height=table_height(len(hist_show)))

st.markdown("---")

# ------------------ Desai Defaults ------------------
st.subheader("🚧 Desai Defaults")
if CSV_TOLLS and not tolls.empty and "tid" in tolls.columns and "tid" in teams.columns:
    tt = tolls.copy()
    if "week" in tt.columns:
        tt = tt[tt["week"].notna()]
        tt = tt[tt["week"] <= MAX_WEEK]
    tt = tt.merge(teams[["tid","team_name","owner"]], on="tid", how="left")
    tt = tt.sort_values(["tid", "week"]).reset_index(drop=True)

    # prior offenses per team BEFORE this row
    cum = tt.groupby("tid")["missing_starters"].cumsum()
    tt["prior_offenses"] = (cum - tt["missing_starters"]).clip(lower=0)

    # per-row penalty and cumulative total
    tt["penalty"] = tt.apply(
        lambda r: compute_toll_row(r["prior_offenses"], r["missing_starters"], TOLL_BASE),
        axis=1
    ).fillna(0.0)
    tt["cumulative_tolls"] = tt.groupby("tid")["penalty"].cumsum().fillna(0.0)

    show_tt = tt.rename(columns={
        "week":"Week", "team_name":"Team", "owner":"Owner", "missing_starters":"Missing Starters",
        "penalty":"Penalty", "cumulative_tolls":"Cumulative Tolls"
    })[["Week","Team","Owner","Missing Starters","Penalty","Cumulative Tolls"]]
    for c in ["Penalty","Cumulative Tolls"]:
        show_tt[c] = show_tt[c].fillna(0).apply(money)

    st.dataframe(show_tt, use_container_width=True, hide_index=True, height=table_height(len(show_tt)))
    st.caption(
        f"Rule: per missing starter offense, penalty doubles: "
        f"{money(TOLL_BASE)}, {money(TOLL_BASE*2)}, {money(TOLL_BASE*4)}, … (cumulative per team)."
    )
else:
    st.info("Add a **Tolls** tab to your Google Sheet and publish its CSV as `CSV_TOLLS`.\nColumns: `week, initials, missing_starters` (or `players_missing`).")

st.markdown("---")

# ------------------ Payouts by Team (ALL teams; Net = Won − Fee − Tolls) ------------------
st.subheader("🏆 Payouts by Team")
awarded = chal.dropna(subset=["winner_tid"]).copy() if "winner_tid" in chal else pd.DataFrame()
if not awarded.empty and "week" in awarded.columns:
    awarded = awarded[awarded["week"].notna()]
    awarded = awarded[awarded["week"] <= MAX_WEEK]

if not awarded.empty:
    totals_won = (awarded.groupby("winner_tid", as_index=False)["prize_amount"].sum()
                         .rename(columns={"winner_tid":"tid","prize_amount":"Total Won"}))
    if "paid" in awarded.columns:
        totals_paid = (awarded[awarded["paid"]].groupby("winner_tid", as_index=False)["prize_amount"].sum()
                         .rename(columns={"winner_tid":"tid","prize_amount":"Total Paid"}))
    else:
        totals_paid = pd.DataFrame({"tid": [], "Total Paid": []})
else:
    totals_won = pd.DataFrame({"tid": [], "Total Won": []})
    totals_paid = pd.DataFrame({"tid": [], "Total Paid": []})

# Tolls totals
if CSV_TOLLS and not tolls.empty and "tid" in tolls.columns:
    tt2 = tolls.copy()
    if "week" in tt2.columns:
        tt2 = tt2[tt2["week"].notna()]
        tt2 = tt2[tt2["week"] <= MAX_WEEK]
    tt2 = tt2.sort_values(["tid","week"]).reset_index(drop=True)

    cum2 = tt2.groupby("tid")["missing_starters"].cumsum()
    tt2["prior_offenses"] = (cum2 - tt2["missing_starters"]).clip(lower=0)
    tt2["penalty"] = tt2.apply(
        lambda r: compute_toll_row(r["prior_offenses"], r["missing_starters"], TOLL_BASE),
        axis=1
    ).fillna(0.0)
    toll_totals = tt2.groupby("tid", as_index=False)["penalty"].sum().rename(columns={"penalty":"Tolls"})
else:
    toll_totals = pd.DataFrame({"tid": [], "Tolls": []})

# Start from ALL teams so zero-winners still show
base = teams[["tid","team_name","owner"]].copy().rename(columns={"team_name":"Team","owner":"Owner"})
by_team = (base
    .merge(totals_won, on="tid", how="left")
    .merge(totals_paid, on="tid", how="left")
    .merge(toll_totals, on="tid", how="left")
)
for c in ["Total Won", "Total Paid", "Tolls"]:
    if c not in by_team.columns: by_team[c] = 0.0
by_team[["Total Won","Total Paid","Tolls"]] = by_team[["Total Won","Total Paid","Tolls"]].fillna(0.0)
by_team["Net"] = by_team["Total Won"] - LEAGUE_FEE - by_team["Tolls"]

by_team = by_team.sort_values(["Total Won","Team"], ascending=[False, True])
disp = by_team[["Team","Owner","Total Won","Total Paid","Tolls","Net"]].copy()

def color_net(s):
    return ["color: green;" if v > 0 else "color: red;" if v < 0 else "" for v in s]

styled = (disp.style
          .format({c: money for c in ["Total Won","Total Paid","Tolls","Net"]})
          .apply(color_net, subset=["Net"]))

st.dataframe(styled, use_container_width=True, hide_index=True, height=table_height(len(disp)))
st.caption(f"Net = Total Won − {money(LEAGUE_FEE)} league fee − Tolls")

# season totals (raw)
raw_tot_won  = float(awarded["prize_amount"].sum()) if not awarded.empty else 0.0
raw_tot_paid = float(awarded.loc[awarded.get("paid", False), "prize_amount"].sum()) if ("paid" in awarded.columns and not awarded.empty) else 0.0
st.markdown("**Season Totals**")
st.write(f"- **Total Awarded:** {money(raw_tot_won)}")
st.write(f"- **Total Paid:** {money(raw_tot_paid)}")

st.markdown("---")

# ------------------ survivor (from Teams) ------------------
st.subheader("🪓 Survivor (Guillotine)")
if "eliminated_week" not in teams.columns:
    st.info("Add an 'eliminated_week' column to Teams (blank = still alive). Optional: eliminated_score, eliminated_note.")
else:
    alive = teams[teams["eliminated_week"].isna()].copy().sort_values("team_name")
    st.markdown(f"**Still Alive ({len(alive)})**")
    alive_view = alive[[c for c in ["team_name","owner"] if c in alive.columns]].rename(columns={
        "team_name":"Team","owner":"Owner"
    })
    st.dataframe(alive_view, use_container_width=True, hide_index=True, height=table_height(len(alive_view)))

    out = teams.dropna(subset=["eliminated_week"]).copy()
    if out.empty:
        st.caption("_No eliminations recorded yet._")
    else:
        out["eliminated_week"] = out["eliminated_week"].astype(int)
        out = out[out["eliminated_week"] <= MAX_WEEK]
        elim_cols = ["eliminated_week","team_name","owner"]
        for c in ["eliminated_score","eliminated_note"]:
            if c in out.columns: elim_cols.append(c)
        out_sorted = out.sort_values(["eliminated_week","team_name"])[elim_cols].rename(columns={
            "eliminated_week":"Week Eliminated",
            "team_name":"Team",
            "owner":"Owner",
            "eliminated_score":"Score",
            "eliminated_note":"Note"
        })
        st.markdown("**Eliminations by Week**")
        st.dataframe(out_sorted, use_container_width=True, hide_index=True, height=table_height(len(out_sorted)))
