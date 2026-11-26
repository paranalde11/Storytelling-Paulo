"""Interactive editorial on electrified vehicle sales in Mexico.
Run with: python -m streamlit run app.py
Data: INEGI RAIAVL hybrid/electric monthly CSVs, 2016–Oct 2026.
"""
from pathlib import Path
import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Mexico's Electric Vehicle Story | Data Desk", page_icon=None, layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent
# Accept both layouts: project/app.py + project/data/ and
# project/code/app (new).py + project/data/.
DATA = next((directory for directory in (ROOT / 'data', ROOT.parent / 'data')
             if directory.is_dir()), ROOT / 'data')
TECH = {'VEH_HIBRIDAS': 'Hybrid (HEV)', 'VEH_HIBRIDAS_PLUGIN': 'Plug-in hybrid (PHEV)', 'VEH_ELECTR': 'Fully electric (BEV)'}
ORDER = list(TECH.values())
COLORS = {ORDER[0]:'#3B82F6', ORDER[1]:'#F59E0B', ORDER[2]:'#14B8A6'}

st.markdown('''<style>
/* Vibrant editorial theme – compatible with Streamlit dark/light surfaces */
:root { --navy:#101E37; --navy2:#1C3154; --blue:#3B82F6; --teal:#14B8A6; --gold:#F59E0B; --coral:#FB7185; --line:rgba(111,151,198,.24); }
.stApp { background:linear-gradient(160deg,#F1F7FF 0%,#F8FBFF 45%,#EFFBF8 100%);color:#1B3150; }
[data-testid="stSidebar"] {background:linear-gradient(180deg,#0D1C35,#18355A 70%,#164B5C);}
[data-testid="stSidebar"] * { color:#EAF5FF!important; }
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p {color:#EAF5FF!important;}
[data-testid="stSidebar"] [data-baseweb="tag"] {background:#327ABD!important;}
.block-container {max-width:1320px;padding-top:2rem;padding-bottom:4rem;}
h1,h2,h3 {letter-spacing:-.035em;color:#10284A!important;}
h1 {font-size:clamp(2rem,4vw,3.25rem)!important;font-weight:850!important;}
h2,h3 {font-weight:760!important;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#FFFFFF,#EAF4FF);border:1px solid #BED8F4;padding:18px 20px;border-radius:16px;box-shadow:0 8px 22px rgba(29,78,136,.09);}
[data-testid="stMetricValue"] {color:#1156B1!important;font-weight:800!important;}
[data-testid="stMetricLabel"] {color:#385676!important;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:7px;padding:7px;background:#E6F0FC;border:1px solid #C6DBF3;border-radius:14px;flex-wrap:wrap;}
[data-testid="stTabs"] [data-baseweb="tab"] {background:transparent;border-radius:9px;color:#294B72!important;font-weight:650;}
[data-testid="stTabs"] [aria-selected="true"] {background:#2358A4!important;color:white!important;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {display:none;}
.stButton button[kind="primary"],.stDownloadButton button {background:#2358A4;color:white;border-radius:10px;border:0;}
.stButton button:hover,.stDownloadButton button:hover {border-color:#14B8A6;color:#14B8A6;}
.eyebrow {color:#117C91;font-weight:900;font-size:.79rem;letter-spacing:.14em;text-transform:uppercase;}
.hero {font-size:1.13rem;line-height:1.75;max-width:990px;color:#395572;background:linear-gradient(100deg,#DAEBFF,#DDF8F0);padding:1.3rem 1.5rem;border-radius:14px;border-left:5px solid #14B8A6;margin-bottom:1.3rem;}
.callout {border-left:5px solid #14B8A6;padding:1rem 1.2rem;background:linear-gradient(90deg,#DDF9F2,#EAF6FF);color:#123C56;border-radius:10px;margin:1rem 0;box-shadow:0 4px 15px rgba(20,184,166,.07);}
.footnote {font-size:.84rem;color:#617A96;}
.story-number {display:inline-block;border:1px solid #3B82F6;border-radius:50px;padding:2px 10px;font-weight:700;margin-right:7px;color:#2358A4;}
[data-testid="stAlert"] {border-radius:12px;}
hr {border-color:#D3E3F4!important;}
@media (max-width: 700px) {.block-container {padding-top:1rem;} .hero {padding:1rem;font-size:1rem;}}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner='Loading INEGI data…')
def load_data():
    files = sorted(DATA.glob('raiavl_hibrido_mensual_tr_cifra_20*.csv'))
    if not files: raise FileNotFoundError('Place annual INEGI CSV files inside the data/ folder.')
    frames=[pd.read_csv(f,dtype={'ID_ENTIDAD':str,'ID_MES':str},encoding='utf-8-sig') for f in files]
    df=pd.concat(frames,ignore_index=True)
    states=pd.read_csv(DATA/'tc_entidad_federativa.csv',dtype={'ID_ENTIDAD':str},encoding='utf-8-sig')
    months=pd.read_csv(DATA/'tc_mes.csv',dtype={'ID_MES':str},encoding='utf-8-sig')
    for d,c in ((df,'ID_ENTIDAD'),(df,'ID_MES'),(states,'ID_ENTIDAD'),(months,'ID_MES')):
        d[c]=d[c].str.zfill(2)
    df['ANIO']=pd.to_numeric(df['ANIO'],errors='raise').astype(int)
    df['date']=pd.to_datetime(dict(year=df.ANIO,month=df.ID_MES.astype(int),day=1))
    for c in TECH:
        df[c]=pd.to_numeric(df[c],errors='raise')
        if df[c].isna().any() or (df[c]<0).any(): raise ValueError(f'Invalid data in {c}')
    if df.duplicated(['date','ID_ENTIDAD']).any(): raise ValueError('Duplicate month/state observations')
    df=df.merge(states[['ID_ENTIDAD','DESC_ENTIDAD']],on='ID_ENTIDAD',how='left',validate='many_to_one')
    df=df.merge(months[['ID_MES','DESCRIPCION_MES']],on='ID_MES',how='left',validate='many_to_one')
    if df[['DESC_ENTIDAD','DESCRIPCION_MES']].isna().any().any():raise ValueError('Missing catalog matches')
    return df.sort_values(['date','ID_ENTIDAD']).reset_index(drop=True)

try: raw=load_data()
except Exception as e:
    st.error(f'Unable to load project data: {e}');st.stop()

latest=raw.date.max()
all_years=sorted(raw.ANIO.unique())
state_names=sorted(raw.loc[raw.ID_ENTIDAD!='99','DESC_ENTIDAD'].unique())

def nfmt(n): return f'{n:,.0f}'
def pfmt(n): return 'N/A' if pd.isna(n) else f'{n:+.1f}%'
def fig_style(fig,height=380,showlegend=True):
    fig.update_layout(template='plotly_white',height=height,margin=dict(l=12,r=25,t=35,b=28),font=dict(color='#1B3150',family='Arial, sans-serif',size=13),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',showlegend=showlegend,legend=dict(orientation='h',y=1.17,x=0,title=''))
    fig.update_xaxes(gridcolor='#DFEAF7',zeroline=False)
    fig.update_yaxes(gridcolor='#DFEAF7',zeroline=False)
    fig.update_layout(hoverlabel=dict(bgcolor='#16345A',font_color='#FFFFFF',font_size=13),legend=dict(orientation='h',y=1.17,x=0,title='',font=dict(color='#27466B')))
    st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False})
def note(txt): st.markdown(f'<div class="callout">{txt}</div>',unsafe_allow_html=True)
def chapter(number,title,sub):
    st.markdown(f'<div class="eyebrow">CHAPTER {number}</div>',unsafe_allow_html=True)
    st.subheader(title);st.write(sub)
def sum_cols(df,cols):return df[cols].sum(axis=1)
def selection_stats(d,columns,focus_year):
    year=d[d.ANIO==focus_year]
    month=int(year.date.dt.month.max())
    current=int(year[columns].sum().sum())
    prior=d[(d.ANIO==focus_year-1)&(d.date.dt.month<=month)]
    months_now=set(year.date.dt.month.unique());months_prior=set(prior.date.dt.month.unique())
    okay=set(range(1,month+1)).issubset(months_now & months_prior)
    prev=int(prior[columns].sum().sum()) if okay else None
    change=(current/prev-1)*100 if prev is not None and prev>0 else np.nan
    return month,current,prev,change

# Navigation is separate from analysis filters. The Overview always tells the
# complete national story; filters are only shown on interactive exploration pages.
PAGES = ['Overview', 'State Explorer', 'Compare Markets', 'Play With the Numbers', 'Data & Methodology']
page = st.sidebar.radio('Sections', PAGES, label_visibility='collapsed')

# The Overview intentionally has no filters and always uses the full data.
year_lo, year_hi = min(all_years), max(all_years)
selected_states = []
selected_tech = ORDER.copy()
include_unknown = True
show_markers = True

if page != 'Overview':
    st.sidebar.markdown('### Explore the data')
    year_lo,year_hi=st.sidebar.slider('Time window',min(all_years),max(all_years),(min(all_years),max(all_years)))
    selected_states=st.sidebar.multiselect('Geography · blank = Mexico',state_names)
    selected_tech=st.sidebar.multiselect('Vehicle technologies',ORDER,default=ORDER)
    include_unknown=st.sidebar.checkbox('Include unspecified geography in national figures',value=True)
    show_markers=st.sidebar.checkbox('Show historical context markers',value=True)
if not selected_tech:st.warning('Please select at least one vehicle technology.');st.stop()
cols=[code for code,name in TECH.items() if name in selected_tech]
base=raw[raw.ANIO.between(year_lo,year_hi)].copy()
if selected_states:base=base[base.DESC_ENTIDAD.isin(selected_states)]
elif not include_unknown:base=base[base.ID_ENTIDAD!='99']
if base.empty:st.warning('No records in this selection.');st.stop()
base['selected_sales']=sum_cols(base,cols)
focus=int(base.ANIO.max())
end_month,focus_value,prev_value,yoy=selection_stats(base,cols,focus)
months_label=pd.Timestamp(focus,end_month,1).strftime('%b')
period_label=f'Jan–{months_label} {focus}'
long=base.melt(id_vars=['date','ANIO','ID_ENTIDAD','DESC_ENTIDAD','ESTATUS'],value_vars=cols,var_name='code',value_name='sales')
long['technology']=long.code.map(TECH)
mix=long[long.ANIO==focus].groupby('technology').sales.sum().reindex(selected_tech,fill_value=0)
leader=mix.idxmax() if mix.sum()>0 else 'None'
leader_share=100*mix.max()/mix.sum() if mix.sum()>0 else np.nan
state_latest=base[(base.ANIO==focus)&(base.ID_ENTIDAD!='99')].groupby('DESC_ENTIDAD').selected_sales.sum().sort_values(ascending=False)
top_state=state_latest.index[0] if len(state_latest) else 'N/A'

st.markdown('<div class="eyebrow">EL FINANCIERO · AUTOMOTIVE DATA DESK · INEGI</div>',unsafe_allow_html=True)
st.title('Mexico’s electric vehicle revolution: a transition in two speeds')
st.markdown('<div class="hero">The market for reported electrified vehicles is expanding. But conventional hybrids remain the largest category, and sales vary sharply across states. Explore the evidence, compare places, and test what these numbers actually mean.</div>',unsafe_allow_html=True)
st.caption(f'INEGI RAIAVL · January 2016–{latest.strftime("%B %Y")} · Current selection: {year_lo}–{year_hi} · Geographic and technology filters apply throughout unless labeled otherwise.')

m1,m2,m3,m4=st.columns(4)
m1.metric('Reported sales · selected years',nfmt(base.selected_sales.sum()))
m2.metric(f'Reported sales · {period_label}',nfmt(focus_value))
m3.metric(f'vs. {focus-1} · same months',pfmt(yoy))
m4.metric('Largest technology share',f'{leader_share:.1f}%' if pd.notna(leader_share) else 'N/A',help=f'{leader}; percent of selected HEV/PHEV/BEV sales, NOT share of all Mexican vehicle sales')
st.caption('Coverage alert: these are sales reported to INEGI for three electrified vehicle technologies, not the entire automotive market. 2026 data end in October.')

if page == 'Overview':
    st.markdown('#### The editorial argument')
    if pd.notna(yoy):
        note(f'For <b>{html.escape(period_label)}</b>, the selected market recorded <b>{nfmt(focus_value)}</b> vehicles, <b>{pfmt(yoy)}</b> versus the same months of {focus-1}. The leading selected technology was <b>{html.escape(leader)}</b> ({leader_share:.1f}%).')
    else:
        note(f'For <b>{html.escape(period_label)}</b>, reported sales reached <b>{nfmt(focus_value)}</b>. A reliable same-month year-over-year rate is unavailable in the selected window. <b>{html.escape(leader)}</b> leads this technology selection.')
    st.caption('All four chapters appear below: growth, technology choices, geographic contrasts, and implications.')
    # Chapter 1: growth
    chapter('01','The rise of electrified vehicle sales','A trailing 12-month view reduces seasonal noise. Lines connect reported monthly values; annotations supply context, not causal proof.')
    monthly=base.groupby('date').selected_sales.sum().sort_index()
    grid=pd.date_range(monthly.index.min(),monthly.index.max(),freq='MS')
    rolling=monthly.reindex(grid).rolling(12,min_periods=12).sum().dropna()
    if not rolling.empty:
        fig=px.line(x=rolling.index,y=rolling.values,labels={'x':'Month','y':'Reported sales · trailing 12 months'})
        fig.update_traces(line=dict(color=COLORS[ORDER[0]],width=4))
        if show_markers:
            for date,label in [('2020-03-01','Pandemic'),('2021-06-01','Chip shortage period'),('2026-01-01','Tariff-policy context')]:
                d=pd.Timestamp(date)
                if rolling.index.min()<=d<=rolling.index.max():
                    x=d.timestamp()*1000;fig.add_vline(x=x,line_dash='dot',line_color='#FB7185');fig.add_annotation(x=x,y=1.02,yref='paper',text=label,showarrow=False,font_size=10)
        fig.update_traces(hovertemplate='%{x|%b %Y}<br>%{y:,.0f} vehicles<extra></extra>')
        fig_style(fig,440,False)
        start=rolling.iloc[0];last=rolling.iloc[-1]
        note(f'The rolling 12-month total moved from <b>{nfmt(start)}</b> to <b>{nfmt(last)}</b> over the selected range. Monthly records alone do not establish what caused the change.')
    else:st.info('Choose at least 12 consecutive months to display the rolling trend.')
    st.markdown('**Why this chapter matters:** It establishes the scale and timing of the transition before asking which technologies are driving it.')
    # Chapter 2: technology
    chapter('02','Not all electrification is the same','HEV, PHEV, and BEV differ substantially. Compare their volumes and shares; the fastest-growing category need not be the largest.')
    same_month=long[long.date.dt.month<=end_month].groupby(['ANIO','technology'],as_index=False).sales.sum()
    fig=px.line(same_month,x='ANIO',y='sales',color='technology',markers=True,color_discrete_map=COLORS,category_orders={'technology':ORDER},labels={'ANIO':'Year','sales':f'Jan–{months_label} sales','technology':'Technology'})
    fig.update_xaxes(dtick=1);fig.update_traces(line_width=3)
    fig_style(fig)
    chart_mix=mix.rename('sales').reset_index().rename(columns={'index':'technology'})
    chart_mix['share']=100*chart_mix.sales/chart_mix.sales.sum() if chart_mix.sales.sum() else 0
    fig=px.bar(chart_mix,x='share',y='technology',orientation='h',color='technology',color_discrete_map=COLORS,text='share',labels={'technology':'','share':'Share of selected technologies (%)'})
    fig.update_traces(texttemplate='%{x:.1f}%',textposition='outside',showlegend=False)
    fig.update_xaxes(range=[0,110]);fig_style(fig,270,False)
    note(f'<b>{html.escape(leader)}</b> holds <b>{leader_share:.1f}%</b> of <b>{html.escape(period_label)}</b> reported hybrid and electric sales in the INEGI dataset.')
    # Chapter 3: geography
    chapter('03','The transition has regional leaders','Sales volume is a measure of market size, not per-capita adoption or charging readiness.')
    state_agg=long[long.ID_ENTIDAD!='99'].groupby(['DESC_ENTIDAD','technology'],as_index=False).sales.sum()
    rank=state_agg.groupby('DESC_ENTIDAD').sales.sum().sort_values(ascending=False)
    if len(rank):
        n=min(10,len(rank))
        picked=rank.head(n).index.tolist()
        sub=state_agg[state_agg.DESC_ENTIDAD.isin(picked)]
        fig=px.bar(sub,x='sales',y='DESC_ENTIDAD',color='technology',orientation='h',barmode='stack',color_discrete_map=COLORS,category_orders={'DESC_ENTIDAD':list(reversed(picked)),'technology':ORDER},labels={'sales':'Reported vehicles sold','DESC_ENTIDAD':'State'})
        fig.update_xaxes(rangemode='tozero');fig_style(fig,max(370,n*39))
        note(f'<b>{html.escape(picked[0])}</b> leads this period by sales volume, at <b>{nfmt(rank.iloc[0])}</b> vehicles. These totals do not account for state population or overall vehicle demand.')
    else:st.info('No named-state observations in the selection.')
    # Chapter 4: implications
    chapter('04','What should the reader take away?','Editorial conclusions should be proportionate to the evidence and distinguish observed demand from unmeasured explanations.')
    if pd.notna(yoy):st.success(f'Same-month reported sales changed by {pfmt(yoy)} in {period_label}.')
    st.markdown('**For buyers:** Do not treat all electrified vehicles as fully electric. **For dealers:** Investigate regional technology preferences before adjusting product mix. **For policymakers:** Pair these figures with charging access, population, and prices before selecting infrastructure investments.')
    st.warning('Reporting coverage is incomplete: some manufacturers, including BYD, are absent. Changes in reporting can distort the appearance of growth. No causal claim about prices, tariffs, or infrastructure is proven here.')
    with st.expander('Create a short editorial briefing'):
        scope='Mexico' if not selected_states else ', '.join(selected_states)
        growth_txt=f'{pfmt(yoy)} against equivalent months of {focus-1}' if pd.notna(yoy) else 'no defensible prior-year comparison in this selection'
        briefing=f'''MEXICO'S ELECTRIFIED VEHICLE STORY — EDITORIAL BRIEF\nScope: {scope}\nPeriod: {period_label}\nReported sales: {nfmt(focus_value)}\nYear-over-year: {growth_txt}\nLeading selected technology: {leader} ({leader_share:.1f}% of selected electrified sales)\nKey message: Electrified vehicle sales and technology choices vary across Mexico. These reported volumes cannot establish total market EV share or explain causation.\nSource: INEGI RAIAVL, 2016–Oct 2026; supplied release revised through October 2026.\n'''
        st.text_area('Generated summary · copy or download',briefing,height=210)
        st.download_button('Download briefing (.txt)',briefing.encode('utf-8'),'editorial_brief.txt','text/plain')

elif page == 'State Explorer':
    st.subheader('State intelligence desk')
    st.write('Select a state to inspect its recent technology mix, trend, and national ranking. This explorer respects the time-window and technology filters, but allows you to pick a state independently of the global geography selection.')
    named=raw[(raw.ANIO.between(year_lo,year_hi))&(raw.ID_ENTIDAD!='99')].copy()
    place=st.selectbox('Profile a state',state_names,index=state_names.index('Nuevo León') if 'Nuevo León' in state_names else 0)
    state_df=named[named.DESC_ENTIDAD==place]
    state_f=state_df[state_df.ANIO==focus]
    state_sales=int(state_f[cols].sum().sum())
    state_all=named[named.ANIO==focus].groupby('DESC_ENTIDAD')[cols].sum().sum(axis=1).sort_values(ascending=False)
    state_rank=int(state_all.index.get_loc(place)+1) if place in state_all.index else None
    e1,e2,e3=st.columns(3)
    e1.metric(f'{place} · {period_label}',nfmt(state_sales))
    e2.metric(f'Rank by selected sales · {focus}',f'{state_rank} / {len(state_all)}' if state_rank else 'N/A')
    e3.metric('Share of named-state sales',f'{100*state_sales/state_all.sum():.1f}%' if state_all.sum() else 'N/A')
    city_long=state_df.melt(id_vars=['date'],value_vars=cols,var_name='code',value_name='sales');city_long['technology']=city_long.code.map(TECH)
    city_mon=city_long.groupby(['date','technology'],as_index=False).sales.sum()
    fig=px.line(city_mon,x='date',y='sales',color='technology',color_discrete_map=COLORS,category_orders={'technology':ORDER},labels={'date':'Month','sales':'Monthly sales'})
    fig_style(fig,380)
    left,right=st.columns([1,1])
    with left:
        state_mix=state_f[cols].sum().rename(index=TECH)
        fig=px.bar(x=state_mix.values,y=state_mix.index,orientation='h',color=state_mix.index,color_discrete_map=COLORS,labels={'x':'Reported vehicles','y':''})
        fig.update_traces(showlegend=False);fig.update_xaxes(rangemode='tozero');fig_style(fig,300,False)
    with right:
        st.markdown('#### Reading the result')
        st.write(f'{place} recorded **{nfmt(state_sales)}** selected electrified-vehicle sales in **{period_label}** and ranks **{state_rank or "N/A"}** by volume among named states.')
        st.info('A large state market may reflect population, vehicle ownership, incomes, or reporting patterns. This dataset cannot isolate these factors.')
    st.download_button('Download this state’s records',state_df.to_csv(index=False).encode('utf-8-sig'),f'{place.replace(" ","_")}_inegi.csv','text/csv')

elif page == 'Compare Markets':
    st.subheader('Compare two state markets')
    st.write('A side-by-side comparison reveals differences in market size and technology composition. It does not establish which state is more advanced per capita.')
    c1,c2=st.columns(2)
    with c1:state_a=st.selectbox('State A',state_names,index=state_names.index('Nuevo León') if 'Nuevo León' in state_names else 0)
    with c2:state_b=st.selectbox('State B',state_names,index=state_names.index('Ciudad de México') if 'Ciudad de México' in state_names else min(1,len(state_names)-1))
    if state_a==state_b:st.warning('Select two different states to compare.')
    else:
        comparisons=raw[(raw.ANIO==focus)&(raw.ID_ENTIDAD!='99')&(raw.DESC_ENTIDAD.isin([state_a,state_b]))]
        pairs=comparisons.groupby('DESC_ENTIDAD')[cols].sum().rename(columns=TECH).reindex([state_a,state_b],fill_value=0)
        totals=pairs.sum(axis=1)
        mm1,mm2=st.columns(2)
        mm1.metric(state_a,nfmt(totals.loc[state_a]));mm2.metric(state_b,nfmt(totals.loc[state_b]))
        method=st.segmented_control('Compare',options=['Sales volume','Technology mix (%)'],default='Sales volume')
        v=pairs if method=='Sales volume' else pairs.div(totals.replace(0,np.nan),axis=0).fillna(0)*100
        stacked=v.reset_index().melt(id_vars='DESC_ENTIDAD',var_name='technology',value_name='value')
        fig=px.bar(stacked,x='DESC_ENTIDAD',y='value',color='technology',barmode='stack',color_discrete_map=COLORS,category_orders={'technology':ORDER},labels={'DESC_ENTIDAD':'State','value':'Vehicles' if method=='Sales volume' else 'Share of selected electrified sales (%)','technology':'Technology'})
        fig.update_yaxes(rangemode='tozero');fig_style(fig,380)
        bigger=totals.idxmax();difference=abs(totals.iloc[0]-totals.iloc[1]);
        note(f'<b>{html.escape(bigger)}</b> recorded more selected sales in <b>{html.escape(period_label)}</b>, a difference of <b>{nfmt(difference)}</b> vehicles. Comparing the percentage mix answers a different question than comparing market size.')
        with st.expander('Exact comparison table'):
            table=pairs.copy();table['Total']=totals
            st.dataframe(table.style.format('{:,.0f}'),use_container_width=True)

elif page == 'Play With the Numbers':
    st.subheader('Explore the Future')
    st.markdown('''What could Mexico’s electrified vehicle market look like if more buyers chose fully electric cars? Change the technology shares and compare your **what-if scenario** with actual reported sales.''')
    st.info('**How to read this page:** This is a learning tool, not a prediction. The reference is reported national sales from January–October 2026, including unspecified geography. The scenario keeps the same total number of vehicles and only changes their technology mix.')

    # Fixed, national reference: using the unfiltered source means the 2026
    # comparison stays valid even if the reader chose different sidebar filters.
    actual_2026 = raw[raw.ANIO == 2026]
    actual = pd.Series({name: float(actual_2026[code].sum()) for code,name in TECH.items()}).reindex(ORDER)
    market_total = float(actual.sum())
    if market_total <= 0:
        st.warning('2026 reference data is unavailable.')
    else:
        actual_pct = actual / market_total * 100
        st.markdown('### 1. Build your own scenario')
        st.write('Move the sliders to assign a hypothetical share to fully electric and plug-in hybrid vehicles. Conventional hybrids automatically receive the remaining percentage so the three shares always add up to 100%.')
        defaults = dict(bev=float(round(actual_pct[ORDER[2]], 1)), phev=float(round(actual_pct[ORDER[1]], 1)))
        left,right=st.columns(2)
        with left:
            bev_share = st.slider('Fully electric (BEV) share (%)', min_value=0.0, max_value=100.0, value=defaults['bev'], step=0.5, key='future_bev_share')
        with right:
            phev_share = st.slider('Plug-in hybrid (PHEV) share (%)', min_value=0.0, max_value=round(100.0-bev_share,1), value=min(defaults['phev'],round(100.0-bev_share,1)), step=0.5, key='future_phev_share')
        hev_share = round(100.0 - bev_share - phev_share, 1)
        pct = pd.Series({ORDER[0]: hev_share, ORDER[1]: phev_share, ORDER[2]: bev_share})
        hypothetical = market_total * pct / 100.0
        c1,c2,c3=st.columns(3)
        c1.metric('Conventional hybrid (HEV)',f'{hev_share:.1f}%',delta=f'{hev_share-actual_pct[ORDER[0]]:+.1f} pp')
        c2.metric('Plug-in hybrid (PHEV)',f'{phev_share:.1f}%',delta=f'{phev_share-actual_pct[ORDER[1]]:+.1f} pp')
        c3.metric('Fully electric (BEV)',f'{bev_share:.1f}%',delta=f'{bev_share-actual_pct[ORDER[2]]:+.1f} pp')

        st.markdown('### 2. See the changes instantly')
        st.write('The chart updates as you move the sliders. Each bar represents the same total number of reported vehicles; only the shares of HEV, PHEV and BEV change.')
        chart_rows=[]
        for label, values in [('Actual Jan–Oct 2026',actual),('Your what-if scenario',hypothetical)]:
            for tech in ORDER:
                chart_rows.append({'Market mix':label,'Technology':tech,'Vehicles':values[tech]})
        chart_df=pd.DataFrame(chart_rows)
        fig=px.bar(chart_df,x='Market mix',y='Vehicles',color='Technology',barmode='stack',category_orders={'Technology':ORDER,'Market mix':['Actual Jan–Oct 2026','Your what-if scenario']},color_discrete_map=COLORS,labels={'Vehicles':'Reported / hypothetical vehicles','Market mix':''},custom_data=['Technology'])
        fig.update_traces(hovertemplate='%{customdata[0]}: %{y:,.0f} vehicles<extra></extra>')
        fig.update_yaxes(rangemode='tozero')
        fig_style(fig,430)
        st.caption('The actual bar is based on INEGI reports. The scenario bar is hypothetical and is not new observed data.')

        st.markdown('### 3. Compare your scenario with the real 2026 market')
        compare=pd.DataFrame({'Technology':ORDER,'Actual sales (Jan–Oct 2026)':[actual[t] for t in ORDER], 'Actual share (%)':[actual_pct[t] for t in ORDER], 'Your scenario sales':[hypothetical[t] for t in ORDER], 'Your scenario share (%)':[pct[t] for t in ORDER]})
        compare['Difference (vehicles)']=compare['Your scenario sales']-compare['Actual sales (Jan–Oct 2026)']
        st.dataframe(compare.style.format({'Actual sales (Jan–Oct 2026)':'{:,.0f}','Actual share (%)':'{:.1f}%','Your scenario sales':'{:,.0f}','Your scenario share (%)':'{:.1f}%','Difference (vehicles)':'{:+,.0f}'}),hide_index=True,use_container_width=True)
        st.caption(f'Reference market total: {market_total:,.0f} reported electrified vehicles sold January–October 2026. The hypothetical total is exactly the same.')

        st.markdown('### 4. What might this mean for Mexico?')
        bev_change=hypothetical[ORDER[2]]-actual[ORDER[2]]
        lead=pct.idxmax()
        if bev_change>1:
            interpretation=f'Your scenario reallocates approximately **{bev_change:,.0f} additional vehicles** to fully electric cars compared with the reported January–October 2026 mix.'
            implications='A shift of this kind could increase the importance of charging access, dealership training and EV-specific after-sales services. These would be questions for further investigation, not proven outcomes of this dataset.'
        elif bev_change < -1:
            interpretation=f'Your scenario allocates approximately **{abs(bev_change):,.0f} fewer vehicles** to fully electric cars than the reported January–October 2026 mix.'
            implications='This would make the conventional or plug-in hybrid categories relatively more important. It is not evidence that consumer demand will actually move this way.'
        else:
            interpretation='Your scenario keeps fully electric vehicle sales close to their reported January–October 2026 level.'
            implications='Try increasing the fully electric share to see how the mix changes while the overall market size stays fixed.'
        st.markdown(interpretation)
        note(f'<b>Largest category in your scenario:</b> {html.escape(lead)} ({pct[lead]:.1f}% of the hypothetical market).')
        st.write(implications)
        st.markdown('**Questions for the reader:** Where would extra charging infrastructure be most useful? Which states currently sell more BEVs? What other data on prices, charging stations and consumers would be needed before making investment decisions?')
        with st.expander('Assumptions and limitations'):
            st.markdown('''- **Not a forecast:** Moving a slider does not estimate future adoption or growth.
- **Fixed volume:** The national total is held at reported January–October 2026 sales; changing shares redistributes that volume.
- **Coverage:** INEGI does not include every brand, so these figures are reported sales, not the entire Mexican auto market.
- **Missing variables:** The scenario does not model prices, consumer budgets, production capacity, charging availability, or policy changes.
- **Time period:** 2026 is only available through October. Do not compare its ten months with a full prior year.''')

elif page == 'Data & Methodology':
    st.subheader('Data transparency and newsroom notes')
    st.markdown('**Source:** INEGI, Registro Administrativo de la Industria Automotriz de Vehículos Ligeros (RAIAVL), reported hybrid and electric sales. Supplied file vintage: revised through October 2026 (metadata modified November 9, 2026).')
    a,b,c=st.columns(3)
    a.metric('Source records',nfmt(len(raw)))
    b.metric('Observed months',raw.date.nunique())
    c.metric('Latest reported month',latest.strftime('%b %Y'))
    st.markdown('#### What the data can and cannot say')
    st.markdown('''- **Measures:** Monthly reported sales of hybrids (HEV), plug-in hybrids (PHEV) and battery-electric vehicles (BEV), by state.
- **Does not measure:** All light-vehicle sales, national EV share of total autos, charging-network coverage, prices, vehicle prices, household incomes, or emissions reductions.
- **Reporting gaps:** Not all automakers report (the assignment specifically mentions BYD); changes in the company reporting panel may make time trends imperfect.
- **Revisions:** Latest numbers are revised or preliminary depending on publication status, and may change.
- **State rankings:** Sales volume does not equal per-capita adoption; records labelled as unspecified are not a geographic state.
- **Events:** COVID-19, the chip shortage, and tariff-policy changes are context annotations, not evidence of causation.
- **Comparability:** For 2026, match January–October with January–October of prior years, never full-year totals.
''')
    st.markdown('#### Reporting status in the current selection')
    st.dataframe(base.ESTATUS.value_counts().rename_axis('INEGI status').reset_index(name='Observations'),hide_index=True,use_container_width=True)
    with st.expander('Data dictionary'):
        st.dataframe(pd.DataFrame({'Column':['ANIO','ID_MES','ID_ENTIDAD','VEH_ELECTR','VEH_HIBRIDAS_PLUGIN','VEH_HIBRIDAS','ESTATUS'], 'Interpretation':['Year','Calendar month ID','State ID','Fully electric vehicles (BEV)','Plug-in hybrids (PHEV)','Conventional hybrids (HEV)','Reporting/revision status']}),hide_index=True,use_container_width=True)
    st.markdown('#### Inspect and export')
    st.dataframe(base[['date','DESC_ENTIDAD']+cols+['ESTATUS']].head(250),hide_index=True,use_container_width=True)
    st.caption('Preview shows the first 250 filtered source rows; download includes all filtered rows.')
    st.download_button('Download all filtered rows (.csv)',base.to_csv(index=False).encode('utf-8-sig'),'inegi_ev_filtered.csv','text/csv')
    st.link_button('Visit INEGI RAIAVL source', 'https://www.inegi.org.mx/datosprimarios/iavl/')

st.divider()
st.caption('Editorial prototype for academic use · Source: INEGI RAIAVL · Design principle: Context → Evidence → Interpretation → Action. Not investment advice.')
