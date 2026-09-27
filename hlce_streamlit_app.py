from __future__ import annotations
import io, math, re, zlib
from itertools import combinations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from scipy.stats import entropy as sp_entropy, hypergeom, kruskal, mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize

st.set_page_config(page_title='Mapa Analítico HLCE', page_icon='🧭', layout='wide')
NAVY='#123B6D'; BLUE='#2673B8'; CYAN='#1AA6A6'; ORANGE='#E66A2C'; BG='#F5FAFD'
st.markdown(f'''<style>
.stApp{{background:{BG}}}.main .block-container{{max-width:1500px;padding-top:1.2rem}}
.hero{{background:linear-gradient(110deg,#0F315D,#18568C 60%,#157E89);border-radius:24px;padding:28px 32px;color:white;box-shadow:0 14px 35px rgba(18,59,109,.18)}}
.hero h1{{color:white;margin:0 0 8px;font-size:2.1rem}}.hero p{{margin:0;opacity:.92}}
.card{{background:white;border:1px solid rgba(38,115,184,.22);border-top:5px solid {BLUE};border-radius:18px;padding:15px;min-height:210px;box-shadow:0 8px 24px rgba(18,59,109,.07)}}
.card.teal{{border-top-color:{CYAN}}}.family{{font-size:.78rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:{NAVY}}}
.card h4{{color:{NAVY};margin:.35rem 0}}.card p{{font-size:.9rem;color:#17324D;line-height:1.35}}.formula{{background:#EEF6FB;border:1px solid #D7EAF5;border-radius:10px;padding:8px;color:{NAVY};font-family:serif;font-size:.88rem}}
.ok,.no{{display:inline-block;margin-top:8px;padding:3px 8px;border-radius:999px;font-size:.72rem;font-weight:700}}.ok{{background:#E8F7F2;color:#136D5D}}.no{{background:#FFF0EA;color:#A84D24}}
div[data-testid='stMetric']{{background:white;border:1px solid #DFEBF3;padding:12px 14px;border-radius:14px}}
</style>''', unsafe_allow_html=True)
st.markdown('''<div class="hero"><h1>Mapa unificado de metodologías analíticas</h1><p>Aplicación interactiva para conversaciones humano–LLM. Todas las cifras y gráficos se calculan exclusivamente sobre el CSV cargado. No se generan observaciones, scores ni datos sintéticos.</p></div>''', unsafe_allow_html=True)

ALIASES={
'message':['message','text','utterance','content','raw_text','response'],
'actor':['actor','speaker','role','author','participant'],
'condition':['condition','group','regime','experimental_condition','arm'],
'turn':['turn','turn_id','turn_index','step','order'],
'conversation_id':['conversation_id','conversation','dialogue_id','session_id','chat_id','thread_id'],
'timestamp':['timestamp','datetime','time','created_at'],
'claim':['claim','statement','hypothesis'],'source':['source','reference','citation']}
HLCE={
'information_flow':['information_flow','info_flow'],'reliability':['reliability'],'durability':['durability','persistence'],'trust':['trust'],
'procedural_transparency':['procedural_transparency','proc_transparency'],'informational_transparency':['informational_transparency','info_transparency'],
'individualization':['individualization','personalization'],'transformation':['transformation']}
WORD_RE=re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_']+",re.UNICODE)
STOP={'de','la','el','y','en','a','los','las','un','una','que','es','por','con','para','del','al','se','lo','como','más','pero','su','sus','le','ya','o','este','esta','esto','si','no','me','te','the','a','an','and','or','of','to','in','is','are','was','were','for','with','on','that','this','it','as','be','by','from','at','i','you','we','they','he','she','not'}

def resolve(df,key):
    low={str(c).lower().strip():c for c in df.columns}
    for a in ALIASES.get(key,[key]):
        if a.lower() in low:return low[a.lower()]
    return None

def resolve_any(df,aliases):
    low={str(c).lower().strip():c for c in df.columns}
    for a in aliases:
        if a.lower() in low:return low[a.lower()]
    return None

def toks(s):return [x.lower() for x in WORD_RE.findall(str(s))]
def safe(a,b):return np.nan if not b or pd.isna(b) else a/b

def normalize_schema(raw):
    m={k:resolve(raw,k) for k in ALIASES}
    if not m['message']: raise ValueError('Falta una columna textual: message/text/utterance/content/raw_text.')
    out=pd.DataFrame(index=raw.index)
    out['message']=raw[m['message']].fillna('').astype(str)
    out['actor']=raw[m['actor']].fillna('DESCONOCIDO').astype(str) if m['actor'] else 'DESCONOCIDO'
    out['condition']=raw[m['condition']].fillna('UNICA').astype(str) if m['condition'] else 'UNICA'
    out['conversation_id']=raw[m['conversation_id']].fillna('conv_1').astype(str) if m['conversation_id'] else 'conv_1'
    if m['turn']:
        out['turn']=pd.to_numeric(raw[m['turn']],errors='coerce')
        out['turn']=out['turn'].fillna(out.groupby('conversation_id').cumcount()+1)
    else: out['turn']=out.groupby('conversation_id').cumcount()+1
    out['timestamp']=pd.to_datetime(raw[m['timestamp']],errors='coerce') if m['timestamp'] else pd.NaT
    for c in raw.columns:
        if c not in out.columns: out[c]=raw[c]
    out['_row']=np.arange(len(out)); out['_tokens']=out['message'].map(toks); out['word_count']=out['_tokens'].map(len)
    out=out.sort_values(['conversation_id','turn','_row']).reset_index(drop=True)
    out['cumulative_words']=out.groupby('conversation_id')['word_count'].cumsum()
    return out,m

@st.cache_resource(show_spinner=False)
def load_sbert(name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)

@st.cache_data(show_spinner=False)
def embed_cached(text_tuple, backend, model_name):
    if backend in ('auto','sbert'):
        try:
            M=load_sbert(model_name); E=M.encode(list(text_tuple),normalize_embeddings=True,show_progress_bar=False)
            return np.asarray(E,float),f'SBERT · {model_name}'
        except Exception as e:
            if backend=='sbert': raise RuntimeError(f'No se pudo ejecutar SBERT: {e}')
    V=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=6000,min_df=1)
    X=V.fit_transform(list(text_tuple)); ns,nf=X.shape
    if ns>=3 and nf>=3:
        nc=max(2,min(64,ns-1,nf-1)); E=TruncatedSVD(n_components=nc,random_state=42).fit_transform(X)
    else:E=X.toarray()
    return normalize(E), 'TF–IDF + SVD'

def getE(df,b,m):return embed_cached(tuple(df['message'].astype(str)),b,m)

def ttr(x):return safe(len(set(x)),len(x))
def mattr(x,w=50):
    if not x:return np.nan
    if len(x)<=w:return ttr(x)
    return float(np.mean([len(set(x[i:i+w]))/w for i in range(len(x)-w+1)]))
def mtld(x,thr=.72):
    if len(x)<10:return np.nan
    def p(seq):
        fac=0.; S=set(); n=0
        for z in seq:
            n+=1;S.add(z);v=len(S)/n
            if v<=thr:fac+=1;S=set();n=0
        if n:fac+=(1-len(S)/n)/max(1e-12,1-thr)
        return len(seq)/fac if fac else np.nan
    return float(np.nanmean([p(x),p(list(reversed(x)))]))
def hdd(x,s=42):
    N=len(x)
    if not N:return np.nan
    d=min(s,N); f=pd.Series(x).value_counts().values
    return float(np.sum([hypergeom.sf(0,N,int(v),d) for v in f])/d)

def lexical_summary(df):
    rows=[]; keys=['condition']+(['actor'] if df['actor'].nunique()>1 else [])
    for k,g in df.groupby(keys,dropna=False):
        if not isinstance(k,tuple):k=(k,)
        x=[t for row in g['_tokens'] for t in row]; r=dict(zip(keys,k))
        r.update(turns=len(g),tokens=len(x),types=len(set(x)),TTR=ttr(x),MATTR_50=mattr(x),MTLD=mtld(x),HDD_42=hdd(x),mean_words_per_turn=g['word_count'].mean())
        rows.append(r)
    return pd.DataFrame(rows)

def distinctive(df,n=15):
    if df['condition'].nunique()<2:return pd.DataFrame()
    g=df.groupby('condition')['message'].apply(lambda s:'\n'.join(s)).sort_index()
    V=TfidfVectorizer(ngram_range=(1,2),max_features=4000,sublinear_tf=True,min_df=1,stop_words=list(STOP))
    X=V.fit_transform(g.values); terms=np.array(V.get_feature_names_out()); rows=[]
    for i,c in enumerate(g.index):
        sc=X[i].toarray().ravel(); ids=np.argsort(sc)[-n:][::-1]
        rows += [dict(condition=c,rank=r+1,term=terms[j],tfidf=float(sc[j])) for r,j in enumerate(ids)]
    return pd.DataFrame(rows)

def semantic_metrics(df,E):
    o=df[['conversation_id','condition','turn','actor','message','word_count']].copy(); n=len(o)
    for c in ['adjacent_similarity','adjacent_distance','semantic_novelty','prior_other_actor_similarity']:o[c]=np.nan
    cents={}
    for c,idx in o.groupby('condition').groups.items():
        v=E[list(idx)].mean(0); cents[c]=v/np.linalg.norm(v) if np.linalg.norm(v) else v
    o['condition_centroid_similarity']=[float(np.dot(E[i],cents[o.loc[i,'condition']])) for i in range(n)]
    for _,idxs in o.groupby('conversation_id').groups.items():
        ids=list(idxs)
        for p,i in enumerate(ids):
            if p:
                j=ids[p-1]; s=float(np.dot(E[i],E[j])); o.loc[i,'adjacent_similarity']=s; o.loc[i,'adjacent_distance']=1-s
                prev=ids[:p]; o.loc[i,'semantic_novelty']=1-float(np.max(E[prev]@E[i]))
                other=[k for k in prev if o.loc[k,'actor']!=o.loc[i,'actor']]
                if other:
                    v=E[other].mean(0); v=v/np.linalg.norm(v) if np.linalg.norm(v) else v
                    o.loc[i,'prior_other_actor_similarity']=float(np.dot(E[i],v))
    return o

def directional(tm):
    rows=[]
    for conv,g in tm.groupby('conversation_id'):
        g=g.sort_values('turn').reset_index(drop=True)
        for i in range(1,len(g)):
            if g.loc[i-1,'actor']!=g.loc[i,'actor'] and pd.notna(g.loc[i,'adjacent_similarity']):
                rows.append(dict(conversation_id=conv,condition=g.loc[i,'condition'],direction=f"{g.loc[i-1,'actor']} → {g.loc[i,'actor']}",similarity=g.loc[i,'adjacent_similarity']))
    if not rows:return pd.DataFrame()
    r=pd.DataFrame(rows)
    return r.groupby(['condition','direction'],as_index=False).agg(mean_similarity=('similarity','mean'),median_similarity=('similarity','median'),n=('similarity','size'))

def matched(df,E):
    if df['condition'].nunique()<2:return pd.DataFrame()
    rows=[]
    for key,g in df.groupby(['turn','actor']):
        cs=list(g['condition'].unique())
        for a,b in combinations(cs,2):
            ia=g.index[g['condition']==a].tolist(); ib=g.index[g['condition']==b].tolist()
            if ia and ib:
                va=normalize(E[ia].mean(0).reshape(1,-1))[0]; vb=normalize(E[ib].mean(0).reshape(1,-1))[0]
                rows.append(dict(turn=key[0],actor=key[1],condition_A=a,condition_B=b,cosine_similarity=float(np.dot(va,vb))))
    return pd.DataFrame(rows)

def info_summary(df):
    rows=[]
    for c,g in df.groupby('condition'):
        x=[t for row in g['_tokens'] for t in row]; txt='\n'.join(g['message'].astype(str)); counts=pd.Series(x).value_counts().values.astype(float) if x else np.array([])
        H=float(sp_entropy(counts/counts.sum(),base=2)) if len(counts) else np.nan
        b=txt.encode('utf-8',errors='ignore'); cr=len(zlib.compress(b,9))/len(b) if b else np.nan
        rows.append(dict(condition=c,turns=len(g),tokens=len(x),shannon_entropy_bits=H,compression_ratio_zlib=cr,lexical_redundancy=1-ttr(x) if x else np.nan))
    return pd.DataFrame(rows)

def divergences(df):
    if df['condition'].nunique()<2:return pd.DataFrame()
    g=df.groupby('condition')['message'].apply(lambda s:'\n'.join(s)).sort_index(); V=CountVectorizer(max_features=3000,stop_words=list(STOP)); X=V.fit_transform(g.values).toarray().astype(float)+1e-9; P=X/X.sum(1,keepdims=True); rows=[]
    for i,j in combinations(range(len(g)),2):
        p,q=P[i],P[j]; m=.5*(p+q)
        rows.append(dict(condition_A=g.index[i],condition_B=g.index[j],KL_A_B_bits=float(np.sum(p*np.log2(p/q))),KL_B_A_bits=float(np.sum(q*np.log2(q/p))),JS_divergence_bits=float(.5*np.sum(p*np.log2(p/m))+.5*np.sum(q*np.log2(q/m)))))
    return pd.DataFrame(rows)

def sem_graph(df,E,thr=.65,max_nodes=250):
    n=min(len(df),max_nodes); S=E[:n]@E[:n].T; G=nx.Graph()
    for i in range(n):G.add_node(i,actor=str(df.loc[i,'actor']),condition=str(df.loc[i,'condition']),turn=float(df.loc[i,'turn']))
    for i in range(n):
        for j in range(i+1,n):
            if S[i,j]>=thr:G.add_edge(i,j,weight=float(S[i,j]))
    deg=nx.degree_centrality(G); bet=nx.betweenness_centrality(G) if G.number_of_edges() else {i:0 for i in G}; comm=[]; Q=np.nan
    if G.number_of_edges():
        try:comm=list(nx.algorithms.community.greedy_modularity_communities(G,weight='weight'));Q=nx.algorithms.community.modularity(G,comm,weight='weight')
        except Exception:pass
    cmap={n:k for k,C in enumerate(comm) for n in C}
    M=pd.DataFrame(dict(node=list(G.nodes),turn=[G.nodes[i]['turn'] for i in G],actor=[G.nodes[i]['actor'] for i in G],condition=[G.nodes[i]['condition'] for i in G],degree_centrality=[deg.get(i,0) for i in G],betweenness=[bet.get(i,0) for i in G],community=[cmap.get(i,-1) for i in G]))
    return G,M,Q

def graph_fig(G,M):
    pos=nx.spring_layout(G,seed=42,weight='weight'); ex=[];ey=[]
    for u,v in G.edges():
        x0,y0=pos[u];x1,y1=pos[v];ex += [x0,x1,None];ey += [y0,y1,None]
    edge=go.Scatter(x=ex,y=ey,mode='lines',line=dict(width=.6,color='rgba(90,110,130,.30)'),hoverinfo='none')
    nxv=[];nyv=[];hover=[];sizes=[]
    for n in G.nodes():
        x,y=pos[n];nxv.append(x);nyv.append(y);r=M[M.node==n].iloc[0];hover.append(f"turn={r.turn}<br>actor={r.actor}<br>condition={r.condition}<br>degree={r.degree_centrality:.3f}");sizes.append(8+26*r.degree_centrality)
    node=go.Scatter(x=nxv,y=nyv,mode='markers',marker=dict(size=sizes,color=M['community'],colorscale='Tealgrn',showscale=True,line=dict(width=.7,color='white')),text=hover,hoverinfo='text')
    fig=go.Figure([edge,node]);fig.update_layout(height=620,title='Grafo de similitud semántica',showlegend=False,xaxis=dict(visible=False),yaxis=dict(visible=False),plot_bgcolor='white',margin=dict(l=10,r=10,t=40,b=10));return fig

def cohend(x,y):
    x=pd.Series(x).dropna().astype(float).values;y=pd.Series(y).dropna().astype(float).values
    if len(x)<2 or len(y)<2:return np.nan
    vx,vy=np.var(x,ddof=1),np.var(y,ddof=1);sp=math.sqrt(((len(x)-1)*vx+(len(y)-1)*vy)/max(1,len(x)+len(y)-2));return (np.mean(x)-np.mean(y))/sp if sp else np.nan

def cliff(x,y):
    x=pd.Series(x).dropna().astype(float).values;y=pd.Series(y).dropna().astype(float).values
    if not len(x) or not len(y):return np.nan
    return (sum(a>b for a in x for b in y)-sum(a<b for a in x for b in y))/(len(x)*len(y))

def compare(df,metric):
    gs={str(c):g[metric].dropna().astype(float).values for c,g in df.groupby('condition') if g[metric].dropna().shape[0]}
    omni=[];pairs=[]
    if len(gs)>=2:
        try:H,p=kruskal(*gs.values());omni=[dict(test='Kruskal–Wallis',statistic=H,p_value=p)]
        except Exception:pass
    for a,b in combinations(gs,2):
        x,y=gs[a],gs[b]
        try:U,p=mannwhitneyu(x,y,alternative='two-sided')
        except Exception:U,p=np.nan,np.nan
        pairs.append(dict(condition_A=a,condition_B=b,mean_A=np.mean(x),mean_B=np.mean(y),delta_mean=np.mean(x)-np.mean(y),cohens_d=cohend(x,y),cliffs_delta=cliff(x,y),mann_whitney_U=U,p_value=p))
    return pd.DataFrame(omni),pd.DataFrame(pairs)

def dl(df,label,name):st.download_button(label,df.to_csv(index=False).encode('utf-8'),file_name=name,mime='text/csv',use_container_width=True)
METHODS=[
('structure','1 · Preparación y estructuración','Integridad, esquema y turn-taking','Valida columnas, vacíos, secuencia de turnos, actores, condiciones y trazabilidad conversacional.','D = {(cᵢ, tᵢ, aᵢ, uᵢ)}ᵢ₌₁ᴺ','blue'),
('literature','2 · Revisión conceptual','Mapa claim ↔ fuente','Solo se ejecuta si el CSV contiene claims y referencias explícitas; no infiere bibliografía ausente.','Aᵢⱼ = 1[claimᵢ ↔ sourceⱼ]','teal'),
('lexical','3 · Análisis léxico-lingüístico','Diversidad, longitud y rasgos textuales','TTR, MATTR, MTLD, HD-D, n-gramas, términos distintivos y distribución de longitud.','TTR = V/N ; MATTR = (1/W) Σ TTR_w','blue'),
('semantic','4 · Semántica computacional','Similitud, novedad y convergencia','Embeddings del texto, similitud coseno, distancia adyacente, novedad y proximidad direccional.','sim(x,y)= eₓ·eᵧ / (||eₓ|| ||eᵧ||)','teal'),
('geometry','5 · Geometría del significado','PCA, centroides y trayectorias','Proyección semántica, centroides por actor/condición, trayectorias y turnos emparejados.','Z = XW ; dᵢⱼ = ||zᵢ-zⱼ||₂','blue'),
('graphs','6 · Estructura relacional y grafos','Red semántica, centralidad y comunidades','Grafo de similitud entre turnos, degree/betweenness, comunidades y modularidad.','Q = (1/2m) Σᵢⱼ(Aᵢⱼ − kᵢkⱼ/2m)δ(cᵢ,cⱼ)','teal'),
('dynamics','7 · Dinámica conversacional','Ritmo, asimetrías y respuesta','Longitud por turno, acumulación, transición de actores, proximidad HUMAN→AI / AI→HUMAN y latencia si existe.','C = (1/(T−1)) Σₜ₌₂ᵀ sim(uₜ,uₜ₋₁)','blue'),
('information','8 · Información y complejidad','Entropía, divergencia y compresión','Shannon, KL/JS entre condiciones, ratio zlib y redundancia léxica.','H(X)=−Σpᵢlog₂pᵢ ; D_KL(P||Q)=Σpᵢlog₂(pᵢ/qᵢ)','teal'),
('coupling','9 · Acoplamiento humano–IA','HLCE / dimensiones observadas','Agrega las ocho dimensiones solo si existen como medidas explícitas en el CSV; no las estima desde texto.','HLCE = Σₖ wₖ zₖ  (solo con zₖ observados)','blue'),
('experiment','10 · Diseño experimental','Comparación entre condiciones','Contrasta métricas mediante Δ, tamaños de efecto y pruebas no paramétricas.','Δ = x̄_A − x̄_B ; d = (x̄_A−x̄_B)/s_p','teal')]

def status(df,id):
    if id in {'structure','lexical','semantic','geometry','graphs','dynamics','information'}:return True,'COMPUTABLE'
    if id=='literature':
        ok=resolve(df,'claim') is not None and resolve(df,'source') is not None;return ok,'COMPUTABLE' if ok else 'Requiere claim + source'
    if id=='coupling':
        ok=all(resolve_any(df,a) is not None for a in HLCE.values());return ok,'COMPUTABLE' if ok else 'Requiere 8 dimensiones HLCE'
    if id=='experiment':
        ok=df['condition'].nunique()>=2;return ok,'COMPUTABLE' if ok else 'Requiere ≥2 condiciones'
    return False,'NO DISPONIBLE'

def render_structure(df,m,b,model):
    st.subheader('1. Preparación y estructuración');st.latex(r'D=\{(c_i,t_i,a_i,u_i)\}_{i=1}^{N}')
    cs=st.columns(4);cs[0].metric('Filas',len(df));cs[1].metric('Conversaciones',df.conversation_id.nunique());cs[2].metric('Actores',df.actor.nunique());cs[3].metric('Condiciones',df.condition.nunique())
    diag=pd.DataFrame({'campo':['message','actor','condition','turn','conversation_id','timestamp'],'columna_origen':[m.get(x) for x in ['message','actor','condition','turn','conversation_id','timestamp']]});st.dataframe(diag,use_container_width=True,hide_index=True)
    dup=df.duplicated(['conversation_id','turn','actor','message']).sum();st.info(f'Duplicados exactos conversación-turno-actor-texto: **{dup}**')
    st.dataframe(df[['conversation_id','condition','turn','actor','message','word_count','cumulative_words']].head(120),use_container_width=True,hide_index=True);dl(df.drop(columns=['_tokens'],errors='ignore'),'Descargar datos normalizados','normalized_conversations.csv')

def render_lit(df,m,b,model):
    st.subheader('2. Revisión conceptual y bibliográfica');st.latex(r'A_{ij}=\mathbf{1}[claim_i\leftrightarrow source_j]')
    c,s=resolve(df,'claim'),resolve(df,'source')
    if not c or not s:st.warning('No computable: faltan claim y source/reference.');return
    L=df[[c,s]].dropna().astype(str).drop_duplicates();st.metric('Enlaces claim ↔ source',len(L));st.dataframe(L.head(250),use_container_width=True,hide_index=True)
    C=L.groupby(c)[s].nunique().sort_values(ascending=False).reset_index(name='n_sources');st.plotly_chart(px.bar(C.head(30),x='n_sources',y=c,orientation='h',title='Fuentes por claim'),use_container_width=True)

def render_lex(df,m,b,model):
    st.subheader('3. Análisis léxico-lingüístico');st.latex(r'\mathrm{TTR}=V/N,\qquad \mathrm{MATTR}=\frac{1}{W}\sum_w \mathrm{TTR}_w')
    S=lexical_summary(df);st.dataframe(S,use_container_width=True,hide_index=True);dl(S,'Descargar métricas léxicas','lexical_metrics.csv')
    st.plotly_chart(px.box(df,x='condition',y='word_count',color='actor',points='outliers',title='Longitud de mensaje por condición y actor'),use_container_width=True)
    T=distinctive(df)
    if not T.empty:
        st.plotly_chart(px.bar(T,x='tfidf',y='term',color='condition',facet_col='condition',facet_col_wrap=2,orientation='h',title='Términos y n-gramas distintivos (TF–IDF)'),use_container_width=True);dl(T,'Descargar términos distintivos','distinctive_lexical_features.csv')

def render_sem(df,m,b,model):
    st.subheader('4. Semántica computacional');st.latex(r'\mathrm{sim}(x,y)=\frac{e_x\cdot e_y}{\|e_x\|\|e_y\|}')
    E,used=getE(df,b,model);T=semantic_metrics(df,E);st.success(f'Backend: {used}')
    cs=st.columns(3);cs[0].metric('Sim. adyacente media',f"{T.adjacent_similarity.mean():.3f}");cs[1].metric('Novedad media',f"{T.semantic_novelty.mean():.3f}");cs[2].metric('Sim. centroide condición',f"{T.condition_centroid_similarity.mean():.3f}")
    st.plotly_chart(px.line(T,x='turn',y='adjacent_distance',color='condition',line_group='conversation_id',markers=True,title='Distancia semántica adyacente'),use_container_width=True)
    st.plotly_chart(px.line(T,x='turn',y='semantic_novelty',color='condition',line_group='conversation_id',markers=True,title='Novedad semántica por turno'),use_container_width=True)
    D=directional(T)
    if not D.empty:st.dataframe(D,use_container_width=True,hide_index=True);st.plotly_chart(px.bar(D,x='direction',y='mean_similarity',color='condition',barmode='group',title='Proximidad direccional'),use_container_width=True);dl(D,'Descargar proximidad direccional','directional_response_proximity.csv')
    dl(T,'Descargar métricas por turno','turn_level_semantic_metrics.csv')

def render_geom(df,m,b,model):
    st.subheader('5. Geometría del significado');st.latex(r'Z=XW,\qquad d_{ij}=\|z_i-z_j\|_2')
    E,used=getE(df,b,model);nc=min(2,E.shape[0],E.shape[1]);Z=PCA(n_components=nc,random_state=42).fit_transform(E) if len(E)>=2 else np.zeros((len(E),1));Z=np.column_stack([Z[:,0],np.zeros(len(Z))]) if Z.shape[1]==1 else Z
    P=df[['condition','actor','turn','conversation_id','message']].copy();P['PC1']=Z[:,0];P['PC2']=Z[:,1];st.caption(f'Entrada semántica: {used}. PCA solo proyecta para visualización.')
    st.plotly_chart(px.scatter(P,x='PC1',y='PC2',color='condition',symbol='actor',hover_data=['turn','conversation_id'],title='Espacio semántico PCA'),use_container_width=True)
    st.plotly_chart(px.line(P.sort_values(['conversation_id','turn']),x='PC1',y='PC2',color='condition',line_group='conversation_id',markers=True,hover_data=['turn','actor'],title='Trayectorias semánticas'),use_container_width=True)
    C=P.groupby(['condition','actor'],as_index=False).agg(PC1=('PC1','mean'),PC2=('PC2','mean'),n=('turn','size'));st.dataframe(C,use_container_width=True,hide_index=True)
    M=matched(df,E)
    if not M.empty:st.dataframe(M.head(300),use_container_width=True,hide_index=True);dl(M,'Descargar matched turns','matched_turn_cross_condition_similarity.csv')
    dl(P,'Descargar PCA','semantic_geometry_pca.csv')
def render_graph(df,m,b,model):
    st.subheader('6. Estructura relacional y grafos');st.latex(r'Q=\frac{1}{2m}\sum_{ij}(A_{ij}-\frac{k_i k_j}{2m})\delta(c_i,c_j)')
    thr=st.slider('Umbral de similitud',.20,.95,.65,.05,key='thr');mx=st.slider('Máximo de turnos',50,500,250,50,key='mx')
    E,used=getE(df,b,model);G,M,Q=sem_graph(df,E,thr,mx);cs=st.columns(3);cs[0].metric('Nodos',G.number_of_nodes());cs[1].metric('Aristas',G.number_of_edges());cs[2].metric('Modularidad','n/d' if pd.isna(Q) else f'{Q:.3f}')
    st.caption(f'Backend: {used}');st.plotly_chart(graph_fig(G,M),use_container_width=True);st.dataframe(M.sort_values('degree_centrality',ascending=False).head(60),use_container_width=True,hide_index=True);dl(M,'Descargar métricas del grafo','semantic_graph_metrics.csv')

def render_dyn(df,m,b,model):
    st.subheader('7. Dinámica conversacional');st.latex(r'C=\frac{1}{T-1}\sum_{t=2}^{T}\mathrm{sim}(u_t,u_{t-1})')
    st.plotly_chart(px.line(df,x='turn',y='word_count',color='actor',line_group='conversation_id',markers=True,title='Longitud por turno'),use_container_width=True)
    st.plotly_chart(px.line(df,x='turn',y='cumulative_words',color='actor',line_group='conversation_id',title='Palabras acumuladas'),use_container_width=True)
    rows=[]
    for conv,g in df.groupby('conversation_id'):
        g=g.sort_values('turn').reset_index(drop=True)
        for i in range(1,len(g)):rows.append(dict(conversation_id=conv,condition=g.loc[i,'condition'],from_actor=g.loc[i-1,'actor'],to_actor=g.loc[i,'actor']))
    if rows:
        R=pd.DataFrame(rows).groupby(['condition','from_actor','to_actor'],as_index=False).size();st.dataframe(R,use_container_width=True,hide_index=True)
    E,used=getE(df,b,model);D=directional(semantic_metrics(df,E))
    if not D.empty:st.markdown('**Proximidad semántica direccional**');st.dataframe(D,use_container_width=True,hide_index=True)
    if df.timestamp.notna().sum()>=2:
        L=df[['conversation_id','condition','turn','actor','timestamp']].copy();L['latency_s']=L.groupby('conversation_id').timestamp.diff().dt.total_seconds();st.plotly_chart(px.box(L.dropna(subset=['latency_s']),x='condition',y='latency_s',color='actor',title='Latencia observada'),use_container_width=True)
    else:st.caption('Latencia no calculada: no hay timestamps suficientes.')

def render_info(df,m,b,model):
    st.subheader('8. Información y complejidad');st.latex(r'H(X)=-\sum_i p_i\log_2p_i,\qquad D_{KL}(P\|Q)=\sum_i p_i\log_2\frac{p_i}{q_i}')
    I=info_summary(df);st.dataframe(I,use_container_width=True,hide_index=True);st.plotly_chart(px.bar(I,x='condition',y=['shannon_entropy_bits','compression_ratio_zlib'],barmode='group',title='Entropía y compresión'),use_container_width=True)
    D=divergences(df)
    if not D.empty:st.dataframe(D,use_container_width=True,hide_index=True);dl(D,'Descargar KL/JS','condition_kl_js_divergence.csv')
    dl(I,'Descargar información y complejidad','condition_information_complexity.csv')

def render_coupling(df,m,b,model):
    st.subheader('9. Psicometría y acoplamiento humano–IA');st.latex(r'\mathrm{HLCE}=\sum_{k=1}^{8}w_k z_k')
    cols={d:resolve_any(df,a) for d,a in HLCE.items()};miss=[d for d,c in cols.items() if c is None]
    if miss:st.warning('No se calcula HLCE. Faltan medidas explícitas: '+', '.join(miss)+'. No se estiman desde texto.');return
    W=pd.DataFrame({'condition':df.condition})
    for d,c in cols.items():W[d]=pd.to_numeric(df[c],errors='coerce')
    Z=W[list(HLCE)].copy()
    for c in Z:
        mu=Z[c].mean();sd=Z[c].std(ddof=0);Z[c]=(Z[c]-mu)/sd if sd and not np.isnan(sd) else np.nan
    cc=st.columns(4);weights={}
    for i,d in enumerate(HLCE):weights[d]=cc[i%4].number_input(d.replace('_',' '),0.0,1.0,1/8,.01,key='w_'+d)
    s=sum(weights.values())
    if s<=0:st.error('La suma de pesos debe ser > 0.');return
    weights={k:v/s for k,v in weights.items()};score=pd.Series(0.,index=W.index);valid=pd.Series(True,index=W.index)
    for d in HLCE:valid &= Z[d].notna();score += weights[d]*Z[d].fillna(0)
    W['HLCE_descriptivo']=score.where(valid,np.nan);S=W.groupby('condition',as_index=False)[list(HLCE)+['HLCE_descriptivo']].mean();st.dataframe(S,use_container_width=True,hide_index=True);st.plotly_chart(px.bar(S,x='condition',y=list(HLCE),barmode='group',title='Dimensiones HLCE observadas'),use_container_width=True);dl(W,'Descargar HLCE observado','hlce_observed_dimensions.csv')
    st.info('El agregado es descriptivo; no presupone validación psicométrica del índice.')

def render_exp(df,m,b,model):
    st.subheader('10. Diseño experimental y comparación de condiciones');st.latex(r'\Delta=\bar{x}_A-\bar{x}_B,\qquad d=\frac{\bar{x}_A-\bar{x}_B}{s_p}')
    if df.condition.nunique()<2:st.warning('Se requieren al menos dos condiciones.');return
    E,used=getE(df,b,model);T=semantic_metrics(df,E);T['word_count']=df.word_count.values
    opts={'Longitud del turno':'word_count','Similitud adyacente':'adjacent_similarity','Distancia adyacente':'adjacent_distance','Novedad semántica':'semantic_novelty','Sim. centroide condición':'condition_centroid_similarity','Proximidad a actor previo':'prior_other_actor_similarity'}
    lab=st.selectbox('Métrica para contrastar',list(opts));metric=opts[lab];O,P=compare(T[['condition',metric]],metric)
    if not O.empty:st.dataframe(O,use_container_width=True,hide_index=True)
    if not P.empty:st.dataframe(P,use_container_width=True,hide_index=True);dl(P,'Descargar comparación','condition_comparison.csv')
    st.plotly_chart(px.box(T,x='condition',y=metric,points='outliers',title=f'{lab} por condición'),use_container_width=True)

RENDER={'structure':render_structure,'literature':render_lit,'lexical':render_lex,'semantic':render_sem,'geometry':render_geom,'graphs':render_graph,'dynamics':render_dyn,'information':render_info,'coupling':render_coupling,'experiment':render_exp}

with st.sidebar:
    st.header('Datos y motor semántico');uploaded=st.file_uploader('Cargar CSV',type=['csv']);sep=st.selectbox('Separador',['Auto',',',';','\\t','|']);backend=st.selectbox('Embeddings',['auto','sbert','tfidf-svd']);model=st.text_input('Modelo SBERT','sentence-transformers/all-MiniLM-L6-v2');st.markdown('---');st.caption('Sin dataset de demostración. Sin imputación de contenido. Si AUTO no dispone de SBERT, TF–IDF + SVD se calcula sobre el propio CSV.')
if uploaded is None:st.info('Carga un CSV. Como mínimo debe incluir message/text/utterance/content/raw_text.');st.stop()
try:
    rawb=uploaded.getvalue();raw=pd.read_csv(io.BytesIO(rawb),sep=None,engine='python') if sep=='Auto' else pd.read_csv(io.BytesIO(rawb),sep='\t' if sep=='\\t' else sep);df,mapping=normalize_schema(raw)
except Exception as e:st.error(f'No se pudo cargar/normalizar el CSV: {e}');st.stop()
cs=st.columns(4);cs[0].metric('Turnos',len(df));cs[1].metric('Conversaciones',df.conversation_id.nunique());cs[2].metric('Actores',df.actor.nunique());cs[3].metric('Condiciones',df.condition.nunique())
st.markdown('## Mapa de métodos');st.caption('Pulsa una tarjeta para calcular ese bloque o ejecuta todos los métodos compatibles con el CSV.')
if 'sel' not in st.session_state:st.session_state.sel=None
if 'all' not in st.session_state:st.session_state.all=False
ca,cb=st.columns(2)
if ca.button('▶ Ejecutar TODO lo computable',type='primary',use_container_width=True):st.session_state.all=True;st.session_state.sel=None
if cb.button('Limpiar resultados',use_container_width=True):st.session_state.all=False;st.session_state.sel=None
for i in range(0,len(METHODS),2):
    cols=st.columns(2)
    for j,M in enumerate(METHODS[i:i+2]):
        id,g,t,d,f,tone=M;ok,msg=status(df,id)
        with cols[j]:
            st.markdown(f'''<div class="card {'teal' if tone=='teal' else ''}"><div class="family">{g}</div><h4>{t}</h4><p>{d}</p><div class="formula">{f}</div><span class="{'ok' if ok else 'no'}">{msg}</span></div>''',unsafe_allow_html=True)
            if st.button('Calcular · '+t,key='run_'+id,use_container_width=True,disabled=not ok):st.session_state.sel=id;st.session_state.all=False
st.markdown('---')
if st.session_state.all:
    st.markdown('## Resultados · ejecución completa')
    for M in METHODS:
        id,g,t,*_=M;ok,msg=status(df,id)
        with st.expander(f'{g} · {t}',expanded=ok):
            if ok:
                try:RENDER[id](df,mapping,backend,model)
                except Exception as e:st.error(f'No pudo completarse: {e}')
            else:st.warning(msg)
elif st.session_state.sel:
    id=st.session_state.sel;t=next(x[2] for x in METHODS if x[0]==id);st.markdown('## Resultado · '+t)
    try:RENDER[id](df,mapping,backend,model)
    except Exception as e:st.error(f'No pudo completarse: {e}')
else:st.info('Selecciona una metodología del mapa para comenzar.')
