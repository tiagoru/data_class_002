# app.py
import os
import io
from datetime import datetime
import sqlite3
import math

import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import pycountry

import re  # add this if not already imported

def normalize_role(text: str) -> str:
    """Trim, collapse spaces, title-case while preserving common acronyms, cap length."""
    s = (text or "").strip()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return ""
    acronyms = {"CEO","CTO","CFO","CMO","COO","CIO","VP","SVP","HR","UX","UI","PM","ML","AI","R&D"}
    tokens = []
    for w in s.split(" "):
        w_up = w.upper()
        tokens.append(w_up if w_up in acronyms else w.capitalize())
    return " ".join(tokens)[:80]  # keep it tidy

# ---------- Optional country→continent mapping ----------
try:
    import country_converter as coco
    _cc = coco.CountryConverter()
except Exception:
    _cc = None

_CONTINENT_BY_ALPHA3 = {
    # minimal fallback coverage (extend as you like)
    "USA":"North America","CAN":"North America","MEX":"North America",
    "BRA":"South America","ARG":"South America","COL":"South America","PER":"South America","CHL":"South America",
    "DEU":"Europe","FRA":"Europe","ESP":"Europe","ITA":"Europe","GBR":"Europe","IRL":"Europe","NLD":"Europe","BEL":"Europe",
    "SWE":"Europe","NOR":"Europe","DNK":"Europe","POL":"Europe","PRT":"Europe","CHE":"Europe","AUT":"Europe","GRC":"Europe",
    "UKR":"Europe","TUR":"Europe",
    "CHN":"Asia","IND":"Asia","JPN":"Asia","KOR":"Asia","SGP":"Asia","MYS":"Asia","THA":"Asia","VNM":"Asia","PHL":"Asia",
    "PAK":"Asia","IDN":"Asia","ARE":"Asia","SAU":"Asia","ISR":"Asia",
    "AUS":"Oceania","NZL":"Oceania",
    "ZAF":"Africa","EGY":"Africa","NGA":"Africa","KEN":"Africa","ETH":"Africa","MAR":"Africa","GHA":"Africa","DZA":"Africa",
}

def alpha3_to_continent(alpha3: str) -> str:
    if not isinstance(alpha3, str):
        return "Other/Unknown"
    a3 = alpha3.upper().strip()
    if _cc:
        try:
            cont = _cc.convert(names=a3, src="ISO3", to="continent")
            if isinstance(cont, list):
                cont = cont[0]
            if cont and cont != "not found":
                return cont
        except Exception:
            pass
    return _CONTINENT_BY_ALPHA3.get(a3, "Other/Unknown")

# ---------- App setup ----------
DB_PATH = os.getenv("DB_PATH", "countries_roles.db")
st.set_page_config(page_title="Audience Survey (Countries & Roles)", page_icon="🌍", layout="wide")

@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            country_name TEXT NOT NULL,
            alpha3 TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)
    return conn

conn = get_conn()

@st.cache_data(ttl=5)
def load_data():
    return pd.read_sql_query("SELECT * FROM responses", conn)

# country list
@st.cache_data
def get_countries():
    rows = []
    for c in pycountry.countries:
        rows.append({"name": c.name, "alpha_3": getattr(c, "alpha_3", None)})
    df = pd.DataFrame(rows).dropna(subset=["alpha_3"]).drop_duplicates("alpha_3")
    df = df.sort_values("name", key=lambda s: s.str.normalize('NFKD'))
    return df

countries_df = get_countries()
country_names = countries_df["name"].tolist()
alpha3_map = dict(zip(countries_df["name"], countries_df["alpha_3"]))



# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📝 Survey", "📄 Data", "📊 Visualizations", "🧠 Insights", "🤖 AI Summary", "📈 Yearly Report"
])


# ---------- Survey ----------
with tab1:
    st.header("Quick survey")
    st.write("Answer the two questions and submit. Results update live.")

    with st.form("survey_form", clear_on_submit=True):
        # Q1: Country (closed list)
        default_idx = country_names.index("United States") if "United States" in country_names else 0
        chosen_country = st.selectbox("1) Where are you from?", country_names, index=default_idx)

        # Q2: Role (open-ended)
        role_input = st.text_input(
            "2) What is your role in your organization?",
            placeholder="e.g., Product Manager / Data Scientist / VP Engineering"
        )

        submitted = st.form_submit_button("Submit response")

    # Handle submission (keep this INSIDE tab1)
    if submitted and chosen_country:
        role_value = normalize_role(role_input)
        if not role_value:
            st.error("Please enter your role.")
        else:
            conn.execute(
                "INSERT INTO responses (ts, country_name, alpha3, role) VALUES (?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), chosen_country, alpha3_map[chosen_country], role_value)
            )
            conn.commit()
            st.success(f"Thanks! Recorded: {chosen_country}, {role_value}")

# ---------- Data ----------
with tab2:
    st.header("Raw data")
    df = load_data()
    if df.empty:
        st.info("No responses yet. Use the Survey tab.")
    else:
        # Countries
        c_counts = (
            df.groupby(["country_name","alpha3"], as_index=False)
              .size().rename(columns={"size":"count"})
              .sort_values("count", ascending=False)
        )
        # Roles
        r_counts = (
            df.groupby(["role"], as_index=False)
              .size().rename(columns={"size":"count"})
              .sort_values("count", ascending=False)
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total responses", int(df.shape[0]))
        with c2:
            st.metric("Unique countries", int(c_counts.shape[0]))
        with c3:
            st.metric("Unique roles", int(r_counts.shape[0]))

        st.subheader("Countries")
        st.dataframe(c_counts, use_container_width=True)

        st.subheader("Roles")
        st.dataframe(r_counts, use_container_width=True)

# ---------- Visualizations ----------
with tab3:
    st.header("Charts")
    df = load_data()
    if df.empty:
        st.info("No data to visualize yet.")
    else:
        # Prepare
        c_counts = (
            df.groupby(["country_name","alpha3"], as_index=False)
              .size().rename(columns={"size":"count"})
              .sort_values("count", ascending=False)
        )
        r_counts = (
            df.groupby(["role"], as_index=False)
              .size().rename(columns={"size":"count"})
              .sort_values("count", ascending=False)
        )

        # --- Countries section ---
        st.markdown("### Countries")
        n_c = len(c_counts)
        max_nc = max(1, min(50, n_c))
        def_nc = min(20, max_nc)
        if n_c <= 1:
            topn_c = n_c
        else:
            topn_c = st.slider("How many countries to show", 1, max_nc, def_nc, 1, key="country_slider")

        bar_c = px.bar(
            c_counts.head(topn_c), x="country_name", y="count",
            labels={"country_name":"Country","count":"Responses"}
        )
        bar_c.update_layout(xaxis_tickangle=-35, margin=dict(l=10,r=10,t=10,b=10), height=400)
        st.plotly_chart(bar_c, use_container_width=True)

        # Word cloud (countries)
        freq_c = {row["country_name"]: int(row["count"]) for _, row in c_counts.iterrows()}
        st.caption("Country frequency word cloud")
        if sum(freq_c.values()) > 0:
            wc = WordCloud(width=1200, height=600, background_color="white")
            wc_img = wc.generate_from_frequencies(freq_c)
            buf = io.BytesIO()
            wc_img.to_image().save(buf, format="PNG")
            st.image(buf.getvalue(), use_column_width=True)

        # World map
        st.subheader("World map")
        map_fig = px.choropleth(
            c_counts, locations="alpha3", color="count",
            hover_name="country_name", color_continuous_scale="Viridis",
            projection="natural earth",
        )
        map_fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=520)
        st.plotly_chart(map_fig, use_container_width=True)

        st.markdown("---")

        # --- Roles section ---
        st.markdown("### Roles")
        n_r = len(r_counts)
        max_nr = max(1, min(30, n_r))
        def_nr = min(15, max_nr)
        if n_r <= 1:
            topn_r = n_r
        else:
            topn_r = st.slider("How many roles to show", 1, max_nr, def_nr, 1, key="role_slider")

        bar_r = px.bar(
            r_counts.head(topn_r), x="role", y="count",
            labels={"role":"Role","count":"Responses"}
        )
        bar_r.update_layout(xaxis_tickangle=-30, margin=dict(l=10,r=10,t=10,b=10), height=400)
        st.plotly_chart(bar_r, use_container_width=True)

        # Word cloud (roles)
        freq_r = {row["role"]: int(row["count"]) for _, row in r_counts.iterrows()}
        st.caption("Role frequency word cloud")
        if sum(freq_r.values()) > 0:
            wc2 = WordCloud(width=1200, height=600, background_color="white")
            wc_img2 = wc2.generate_from_frequencies(freq_r)
            buf2 = io.BytesIO()
            wc_img2.to_image().save(buf2, format="PNG")
            st.image(buf2.getvalue(), use_column_width=True)

        # Optional: Roles by continent (stacked bar)
        dfc = df.copy()
        dfc["continent"] = dfc["alpha3"].apply(alpha3_to_continent)
        cross = (dfc.groupby(["continent","role"]).size()
                 .reset_index(name="count"))
        if not cross.empty:
            st.subheader("Roles by continent")
            stacked = px.bar(cross, x="continent", y="count", color="role", barmode="stack")
            stacked.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=520, legend_traceorder="reversed")
            st.plotly_chart(stacked, use_container_width=True)

# ---------- Insights (separate for Countries & Roles) ----------
with tab4:
    st.header("Statistical Insights")
    df = load_data()
    if df.empty:
        st.info("No data yet. Insights will appear once there are responses.")
    else:
        # helpers
        def wilson_ci(k, n, z=1.96):
            if n == 0: return (0.0, 0.0)
            p = k / n
            denom = 1 + z**2 / n
            center = (p + z**2/(2*n)) / denom
            half = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
            return (max(0.0, center - half), min(1.0, center + half))

        def two_prop_z(k1, n1, k2, n2):
            if min(n1, n2) == 0: return None, None
            p1, p2 = k1/n1, k2/n2
            p_pool = (k1 + k2) / (n1 + n2)
            se = math.sqrt(p_pool*(1-p_pool) * (1/n1 + 1/n2))
            if se == 0: return None, None
            z = (p1 - p2) / se
            from math import erf, sqrt
            def cdf(x): return 0.5 * (1 + erf(x / sqrt(2)))
            pval = 2 * (1 - cdf(abs(z)))
            return z, pval

        # Countries
        st.subheader("Countries")
        c_counts = (df.groupby(["country_name"], as_index=False)
                      .size().rename(columns={"size":"count"})
                      .sort_values("count", ascending=False))
        total = int(df.shape[0])
        unique_c = int(c_counts.shape[0])
        st.write(f"Sample size: **n = {total}** | Unique countries: **{unique_c}**")

        if not c_counts.empty:
            top = c_counts.iloc[0]
            top_country, top_k = top["country_name"], int(top["count"])
            share = top_k / total
            lo, hi = wilson_ci(top_k, total)
            lines_c = [f"- **Most represented country:** {top_country} — {top_k}/{total} ({share:.1%}) [95% CI {lo:.1%}–{hi:.1%}]."]
            if len(c_counts) >= 2:
                second = c_counts.iloc[1]
                z2, p2 = two_prop_z(top_k, total, int(second['count']), total)
                if z2 is not None:
                    if p2 < 0.05:
                        lines_c.append(f"- **Lead over runner-up:** significant (z={z2:.2f}, p={p2:.3f}).")
                    else:
                        lines_c.append(f"- **Lead over runner-up:** not significant (z={z2:.2f}, p={p2:.3f}).")
            # Diversity across countries
            p = c_counts["count"] / total
            hhi = float((p**2).sum()); diversity = (1/hhi) if hhi>0 else 0
            if diversity >= 8:
                lines_c.append("- **Country diversity:** high (broad distribution).")
            elif diversity >= 4:
                lines_c.append("- **Country diversity:** moderate (a few standouts).")
            else:
                lines_c.append("- **Country diversity:** low (concentrated).")
            st.markdown("\n".join(lines_c))

        st.markdown("---")

        # Roles
        st.subheader("Roles")
        r_counts = (df.groupby(["role"], as_index=False)
                      .size().rename(columns={"size":"count"})
                      .sort_values("count", ascending=False))
        unique_r = int(r_counts.shape[0])
        st.write(f"Unique roles: **{unique_r}**")

        if not r_counts.empty:
            top_r = r_counts.iloc[0]
            top_role, top_rk = top_r["role"], int(top_r["count"])
            r_share = top_rk / total
            lo_r, hi_r = wilson_ci(top_rk, total)
            lines_r = [f"- **Most common role:** {top_role} — {top_rk}/{total} ({r_share:.1%}) [95% CI {lo_r:.1%}–{hi_r:.1%}]."]
            if len(r_counts) >= 2:
                second_r = r_counts.iloc[1]
                z2, p2 = two_prop_z(top_rk, total, int(second_r['count']), total)
                if z2 is not None:
                    if p2 < 0.05:
                        lines_r.append(f"- **Lead over next role:** significant (z={z2:.2f}, p={p2:.3f}).")
                    else:
                        lines_r.append(f"- **Lead over next role:** not significant (z={z2:.2f}, p={p2:.3f}).")
            # Diversity across roles
            pr = r_counts["count"] / total
            hhi_r = float((pr**2).sum()); diversity_r = (1/hhi_r) if hhi_r>0 else 0
            if diversity_r >= 8:
                lines_r.append("- **Role diversity:** high (many roles represented).")
            elif diversity_r >= 4:
                lines_r.append("- **Role diversity:** moderate.")
            else:
                lines_r.append("- **Role diversity:** low (few roles dominate).")
            st.markdown("\n".join(lines_r))

# ---------- AI Summary (continents + roles) ----------
with tab5:
    st.header("AI Audience Summary (Continents + Roles)")
    df = load_data()
    if df.empty:
        st.info("No data yet.")
    else:
        # Continents
        dfc = df.copy()
        dfc["continent"] = dfc["alpha3"].apply(alpha3_to_continent)
        cont = (dfc.groupby("continent", as_index=False).size()
                    .rename(columns={"size":"count"})
                    .sort_values("count", ascending=False))
        known = cont[cont["continent"]!="Other/Unknown"].copy()
        known_total = int(known["count"].sum()) if not known.empty else 0
        if known_total > 0:
            known["share"] = known["count"] / known_total
        # Roles
        r_counts = (df.groupby(["role"], as_index=False)
                      .size().rename(columns={"size":"count"})
                      .sort_values("count", ascending=False))

        total = int(df.shape[0])
        top_cont_breakdown = ", ".join(
            f"{row.continent} ({int(row['count'])}, {row['share']:.0%})" if "share" in row else f"{row.continent} ({int(row['count'])})"
            for _, row in known.head(5).iterrows()
        ) if known_total else "—"
        top_roles = ", ".join(f"{row.role} ({int(row['count'])})" for _, row in r_counts.head(5).iterrows()) if not r_counts.empty else "—"

        # Controls
        c1, c2 = st.columns(2)
        with c1:
            tone = st.selectbox("Tone", ["Professional", "Friendly", "Energetic", "Neutral"], index=0)
        with c2:
            paragraphs = st.radio("Length", ["1 paragraph", "2 paragraphs"], index=1)

        # Local fallback
        def local_summary():
            lead = f"We've gathered {total} responses across multiple regions." if known_total else f"We've gathered {total} responses."
            p1 = (
                f"{lead} Top continental representation: {top_cont_breakdown}. "
                f"On roles, leading functions include {top_roles}."
            )
            p2 = "We’ll keep tracking how the regional and functional mix evolves as more participants join." \
                 if paragraphs == "2 paragraphs" else ""
            return p1 if not p2 else p1 + "\n\n" + p2

        # Generate with OpenAI (optional)
        summary = ""
        if st.button("Generate summary"):
            if os.getenv("OPENAI_API_KEY"):
                try:
                    from openai import OpenAI
                    client = OpenAI()
                    length_req = "one short paragraph (~80 words)" if paragraphs == "1 paragraph" else "two short paragraphs (100–160 words total)"
                    prompt = (
                        "Write {len_req} summarizing a survey audience by continent and role. "
                        "Be {tone}, factual, concise; no emojis, hashtags, or URLs.\n\n"
                        f"Facts:\n- Total responses: {total}\n- Continental breakdown (top): {top_cont_breakdown}\n"
                        f"- Roles (top): {top_roles}\n"
                    ).format(len_req=length_req, tone=tone.lower())

                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"user","content":prompt}],
                        temperature=0.7,
                        max_tokens=450,
                    )
                    summary = resp.choices[0].message.content.strip()
                except Exception as e:
                    st.warning(f"OpenAI generation failed: {e}. Showing local summary instead.")
                    summary = local_summary()
            else:
                st.info("OPENAI_API_KEY not set — using local summary.")
                summary = local_summary()

        if summary:
            st.subheader("Audience summary")
            st.markdown(summary)
            st.download_button(
                "Download summary (.txt)",
                data=summary.encode("utf-8"),
                file_name="audience_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )
# ---------- Yearly Report (KPIs, YoY deltas, cross-tabs, trends) ----------
with tab6:
    st.header("Yearly Report")

    df_all = load_data()
    if df_all.empty:
        st.info("No data yet.")
    else:
        # ---- Prepare time & geography ----
        dft = df_all.copy()
        dft["ts"] = pd.to_datetime(dft["ts"], utc=True, errors="coerce")
        dft = dft.dropna(subset=["ts"])
        dft["year"] = dft["ts"].dt.year
        dft["month"] = dft["ts"].dt.to_period("M").astype(str)

        # Map continents
        dft["continent"] = dft["alpha3"].apply(alpha3_to_continent)

        # Year selector (defaults to most recent)
        years = sorted(dft["year"].unique().tolist())
        default_year = years[-1]
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            year_sel = st.selectbox("Report year", years, index=years.index(default_year))
        with c2:
            compare_prev = st.checkbox("Compare against previous year", value=True)
        with c3:
            topn_countries_heatmap = st.slider("Top-N countries for country×role heatmap",
                                               min_value=5, max_value=20, value=10, step=1)

        cur = dft[dft["year"] == year_sel].copy()
        prev = dft[dft["year"] == (year_sel - 1)].copy() if compare_prev else pd.DataFrame(columns=dft.columns)

        # Helpers
        def wilson_ci(k, n, z=1.96):
            if n == 0: return (0.0, 0.0)
            p = k / n
            denom = 1 + z**2 / n
            center = (p + z**2/(2*n)) / denom
            half = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
            return (max(0.0, center - half), min(1.0, center + half))

        def inv_hhi(shares):
            shares = [s for s in shares if s > 0]
            return (1 / sum(s*s for s in shares)) if shares else 0

        # ---- KPIs ----
        total = int(len(cur))
        c_counts = (cur.groupby(["country_name","alpha3"], as_index=False)
                      .size().rename(columns={"size":"count"})
                      .sort_values("count", ascending=False))
        r_counts = (cur.groupby("role", as_index=False)
                      .size().rename(columns={"size":"count"})
                      .sort_values("count", ascending=False))
        cont_counts = (cur[cur["continent"]!="Other/Unknown"]
                       .groupby("continent", as_index=False).size()
                       .rename(columns={"size":"count"})
                       .sort_values("count", ascending=False))

        unique_countries = int(c_counts["country_name"].nunique())
        continents_covered = int(cont_counts["continent"].nunique())

        # Top country KPI
        if not c_counts.empty:
            top_country = c_counts.iloc[0]
            top_country_share = top_country["count"] / total if total else 0
            lo, hi = wilson_ci(int(top_country["count"]), total)
        else:
            top_country = {"country_name":"—","count":0}
            top_country_share, lo, hi = 0, 0, 0

        # Top role KPI
        if not r_counts.empty:
            top_role = r_counts.iloc[0]
            top_role_share = top_role["count"] / total if total else 0
            lo_r, hi_r = wilson_ci(int(top_role["count"]), total)
        else:
            top_role = {"role":"—","count":0}
            top_role_share, lo_r, hi_r = 0, 0, 0

        # Regional Balance Index (inverse HHI across continents)
        cont_total = cont_counts["count"].sum()
        cont_shares = (cont_counts["count"] / cont_total).tolist() if cont_total else []
        rbi = inv_hhi(cont_shares)

        # Role Diversity Score (inverse HHI across roles)
        role_total = r_counts["count"].sum()
        role_shares = (r_counts["count"] / role_total).tolist() if role_total else []
        role_div = inv_hhi(role_shares)

        # Top-5 countries coverage
        top5_coverage = (c_counts["count"].head(5).sum() / total) if total else 0

        # KPI display
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total responses", total)
        k2.metric("Unique countries", unique_countries)
        k3.metric("Continents covered", continents_covered)
        k4.metric("Top country share",
                  f"{top_country_share:.1%}",
                  f"[{lo:.0%}–{hi:.0%}]")
        k5.metric("Top role share",
                  f"{top_role_share:.1%}",
                  f"[{lo_r:.0%}–{hi_r:.0%}]")
        k6.metric("Top-5 countries coverage", f"{top5_coverage:.1%}")
        st.caption(f"Regional Balance Index (continents): **{rbi:.2f}**  |  Role Diversity Score: **{role_div:.2f}**")

        # ---- YoY deltas ----
        st.subheader("Year-over-year changes")
        if not prev.empty and len(prev) > 0:
            # Continent shares YoY
            cur_cont = cont_counts.copy()
            cur_cont["share"] = cur_cont["count"] / cont_total if cont_total else 0

            prev_cont = (prev[prev["continent"]!="Other/Unknown"].groupby("continent", as_index=False)
                         .size().rename(columns={"size":"count"}))
            prev_total = prev_cont["count"].sum()
            prev_cont["share"] = prev_cont["count"] / prev_total if prev_total else 0

            yoy_cont = pd.merge(cur_cont[["continent","share"]],
                                prev_cont[["continent","share"]],
                                on="continent", how="outer", suffixes=("_cur","_prev")).fillna(0)
            yoy_cont["Δpp"] = (yoy_cont["share_cur"] - yoy_cont["share_prev"]) * 100

            cont_bar = px.bar(yoy_cont.sort_values("Δpp", ascending=False),
                              x="continent", y="Δpp",
                              title=None, labels={"Δpp":"Δ (pp)", "continent":"Continent"})
            cont_bar.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=350)
            st.plotly_chart(cont_bar, use_container_width=True)

            # Top roles YoY (only roles appearing this year)
            cur_roles = r_counts.copy()
            cur_roles["share"] = cur_roles["count"] / role_total if role_total else 0
            prev_roles = (prev.groupby("role", as_index=False).size()
                          .rename(columns={"size":"count"}))
            prev_roles_total = prev_roles["count"].sum()
            prev_roles["share"] = prev_roles["count"] / prev_roles_total if prev_roles_total else 0

            yoy_roles = pd.merge(cur_roles[["role","share"]],
                                 prev_roles[["role","share"]],
                                 on="role", how="left", suffixes=("_cur","_prev")).fillna(0)
            yoy_roles["Δpp"] = (yoy_roles["share_cur"] - yoy_roles["share_prev"]) * 100
            yoy_roles = yoy_roles.sort_values("Δpp", ascending=False).head(12)

            role_bar = px.bar(yoy_roles, x="role", y="Δpp",
                              labels={"role":"Role","Δpp":"Δ (pp)"})
            role_bar.update_layout(xaxis_tickangle=-25, margin=dict(l=10,r=10,t=10,b=10), height=350)
            st.plotly_chart(role_bar, use_container_width=True)
        else:
            st.info("No previous year to compare.")

        st.markdown("---")

        # ---- Cross-tabs ----
        st.subheader("Continent × Role (heatmap)")
        cross_cr = (cur[cur["continent"]!="Other/Unknown"].groupby(["continent","role"])
                    .size().reset_index(name="count"))
        if not cross_cr.empty:
            # pivot for heatmap
            pivot_cr = cross_cr.pivot(index="continent", columns="role", values="count").fillna(0)
            hm1 = px.imshow(pivot_cr.values,
                            x=pivot_cr.columns, y=pivot_cr.index,
                            color_continuous_scale="Viridis",
                            labels=dict(x="Role", y="Continent", color="Count"))
            hm1.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=520)
            st.plotly_chart(hm1, use_container_width=True)
        else:
            st.info("Not enough data for continent × role.")

        st.subheader("Top countries × Role (heatmap)")
        top_c = c_counts.head(topn_countries_heatmap)["country_name"].tolist()
        cross_cr2 = (cur[cur["country_name"].isin(top_c)]
                     .groupby(["country_name","role"]).size().reset_index(name="count"))
        if not cross_cr2.empty:
            piv2 = cross_cr2.pivot(index="country_name", columns="role", values="count").fillna(0)
            hm2 = px.imshow(piv2.values,
                            x=piv2.columns, y=piv2.index,
                            color_continuous_scale="Viridis",
                            labels=dict(x="Role", y="Country", color="Count"))
            hm2.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=520)
            st.plotly_chart(hm2, use_container_width=True)
        else:
            st.info("Not enough data for country × role.")

        st.markdown("---")

        # ---- Trends ----
        st.subheader("Monthly submissions")
        by_month = (cur.groupby("month").size().reset_index(name="count")
                    .sort_values("month"))
        if not by_month.empty:
            line = px.line(by_month, x="month", y="count", markers=True,
                           labels={"month":"Month","count":"Submissions"})
            line.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=320)
            st.plotly_chart(line, use_container_width=True)

        st.subheader("Monthly submissions by continent (stacked area)")
        by_month_cont = (cur[cur["continent"]!="Other/Unknown"]
                         .groupby(["month","continent"]).size()
                         .reset_index(name="count")
                         .sort_values("month"))
        if not by_month_cont.empty:
            area = px.area(by_month_cont, x="month", y="count", color="continent",
                           groupnorm=None, labels={"count":"Submissions","month":"Month"})
            area.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=380)
            st.plotly_chart(area, use_container_width=True)

        # Momentum: last 3 months vs previous 3 months by continent
        st.subheader("Momentum (last 3 months vs previous 3) by continent")
        if not by_month_cont.empty:
            # Order months and pick windows
            months_sorted = sorted(by_month_cont["month"].unique())
            last3 = months_sorted[-3:] if len(months_sorted) >= 3 else months_sorted
            prev3 = months_sorted[-6:-3] if len(months_sorted) >= 6 else []
            m_last = by_month_cont[by_month_cont["month"].isin(last3)].groupby("continent")["count"].sum()
            m_prev = by_month_cont[by_month_cont["month"].isin(prev3)].groupby("continent")["count"].sum()
            momentum = pd.DataFrame({
                "continent": sorted(set(m_last.index).union(set(m_prev.index))),
            })
            momentum["last_3"] = momentum["continent"].map(m_last).fillna(0).astype(int)
            momentum["prev_3"] = momentum["continent"].map(m_prev).fillna(0).astype(int)
            momentum["Δ"] = momentum["last_3"] - momentum["prev_3"]
            st.dataframe(momentum.sort_values("Δ", ascending=False), use_container_width=True)

        st.markdown("---")

        # ---- Equity & concentration checks ----
        st.subheader("Balance & concentration checks")
        if not cont_counts.empty:
            shares = (cont_counts.set_index("continent")["count"] / cont_total)
            max_share = shares.max() if not shares.empty else 0
            min_share = shares.min() if not shares.empty else 0
            st.write(f"- **Max–min continent share gap:** {(max_share - min_share)*100:.1f} pp")
        # Top-3 role concentration per continent
        cross_roles = (cur[cur["continent"]!="Other/Unknown"]
                       .groupby(["continent","role"]).size().reset_index(name="count"))
        if not cross_roles.empty:
            totals_by_cont = cross_roles.groupby("continent")["count"].sum()
            top3_conc = (cross_roles.sort_values(["continent","count"], ascending=[True,False])
                         .groupby("continent").head(3).groupby("continent")["count"].sum()
                        ) / totals_by_cont
            conc_df = top3_conc.reset_index().rename(columns={"count":"Top-3 role concentration"})
            conc_df["Top-3 role concentration"] = (conc_df["Top-3 role concentration"] * 100).round(1)
            st.dataframe(conc_df, use_container_width=True)

        # ---- Downloads ----
        st.subheader("Download data")
        # Build CSVs
        out_country = c_counts.copy()
        out_roles = r_counts.copy()
        out_cont = cont_counts.copy()
        st.download_button("Download country counts (CSV)",
                           data=out_country.to_csv(index=False).encode("utf-8"),
                           file_name=f"country_counts_{year_sel}.csv",
                           mime="text/csv", use_container_width=True)
        st.download_button("Download role counts (CSV)",
                           data=out_roles.to_csv(index=False).encode("utf-8"),
                           file_name=f"role_counts_{year_sel}.csv",
                           mime="text/csv", use_container_width=True)
        st.download_button("Download continent counts (CSV)",
                           data=out_cont.to_csv(index=False).encode("utf-8"),
                           file_name=f"continent_counts_{year_sel}.csv",
                           mime="text/csv", use_container_width=True)
