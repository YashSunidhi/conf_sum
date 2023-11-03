import streamlit as st
import pandas as pd
import datetime as datetime
import os
import ast
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
import plotly.express as px

from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import PartOfSpeech, KeyBERTInspired, MaximalMarginalRelevance, OpenAI
import streamlit as st
from trubrics.integrations.streamlit import FeedbackCollector
from streamlit_feedback import streamlit_feedback
from streamlit_extras.stylable_container import stylable_container

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: black;'> Conference Listening GenAI </h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: black;'> Understanding Discussion, Observation and Opinions </h5>", unsafe_allow_html=True)
# uploaded_file = st.sidebar.file_uploader("Upload a Summary csv File",type= 'csv' , key="file")
# uploaded_file1 = st.sidebar.file_uploader("Upload a Concept csv File",type= 'csv' , key="file1")
# # uploaded_file2 = st.sidebar.file_uploader("Upload a Summary NR csv File",type= 'csv' , key="file2")
# if uploaded_file is not None:
#     if uploaded_file1 is not None:
#         # if uploaded_file2 is not None:
def main_page():
    #create_space(1)
    # with st.expander(label="# To Compare different Summarization and Provide feedback on what good will look like", expanded=False):
    #     st.markdown("""

    #     1) Upload Summary of Representative Document

    #     2) Upload Document Based Concept Modelled File 

    #     3) Upload Summary of Non- Representative Document

    #     4) Observe and Provide Comment
    #     """)
    # "---"

    #st.markdown("<h3 style='text-align: center; color: grey;'> Theme Based Summary Review </h3>", unsafe_allow_html=True)
    #col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
    
    df = pd.read_csv('./test_conf_ectrims_2023_18_oct/Cluster_Summary_18_Oct_final.csv')
    df1 = pd.read_csv('./test_conf_ectrims_2023_18_oct/Integrated_Outcome_18_oct.csv')
    #df['User_Input'] = ''
    #df1 = pd.read_csv(uploaded_file1)
    # df2 = pd.read_csv(uploaded_file2)
    concept_option = st.sidebar.selectbox(
    'Theme Selection',
    (df['Concept'].unique()))
    
    collector = FeedbackCollector(
        email='smnitrkl50@gmail.com',
        password='Ram@2107',
        project="default"
    )
    #st.write(f"Identified Theme: **{concept_option}**")
    #st.markdown("<h5 style='text-align: center; color: grey;'> Representative Docs base Summary (Problem-Solution Structure) </h5>", unsafe_allow_html=True)
    do1 = df[(df['Concept']==concept_option)]['Summary'].reset_index(drop=True)[0]
    do2 = df[(df['Concept']==concept_option)]['Title'].reset_index(drop=True)[0]
    do3 = ' \n\n'.join(df[(df['Concept']==concept_option)]['Represented Document'].reset_index(drop=True).to_list())
    # do3_1 = ' \n\n'.join(df1[(df1['CustomName']==concept_option) & (df1['Sentiment_label']=='positive') ]['Full Text'].reset_index(drop=True).to_list())
    # do3_2 = ' \n\n'.join(df1[(df1['CustomName']==concept_option) & (df1['Sentiment_label']=='negative') ]['Full Text'].reset_index(drop=True).to_list())
    # do3_3 = ' \n\n'.join(df1[(df1['CustomName']==concept_option) & (df1['Sentiment_label']=='neutral') ]['Full Text'].reset_index(drop=True).to_list())
    do4 = ', '.join(df[(df['Concept']==concept_option)]['Keywords_u'].reset_index(drop=True).to_list())
    st.markdown("<h4 style='text-align: center; color: grey;'> Theme Title </h4>", unsafe_allow_html=True)
    st.write(do2)
    st.markdown("<h4 style='text-align: center; color: grey;'> Theme Summary </h4>", unsafe_allow_html=True)
    st.write(do1)
    st.markdown("<h4 style='text-align: center; color: grey;'> Reference Posts </h4>", unsafe_allow_html=True)
    st.write(do3)
    # cola, colb, colc = st.columns([1,1,1])
    # cola.markdown("<h6 style='text-align: center; color: grey;'> Reference Posts with Positive Sentiments </h6>", unsafe_allow_html=True)
    # cola.write(do3_1)
    # colb.markdown("<h6 style='text-align: center; color: grey;'> Reference Posts with Negative Sentiments </h6>", unsafe_allow_html=True)
    # colb.write(do3_2)
    # colc.markdown("<h6 style='text-align: center; color: grey;'> Reference Posts with Neutral Sentiments </h6>", unsafe_allow_html=True)
    # colc.write(do3_3)
    st.markdown("<h6 style='text-align: center; color: grey;'> Keywords Identified </h6>", unsafe_allow_html=True)
    st.write(do4)

def concept_view_1():
   
    df = pd.read_csv('./test_conf_ectrims_2023_18_oct/Integrated_Outcome_18_oct.csv')
    df = df.fillna('NA')
    abstracts = df['Full Text'].to_list()
    titles = df["Title"].to_list()
    #df['Document'] = df['clean_tw']


    # dx = pd.read_csv('./test_conf_ectrims_2023/document_cluster_ectrims_2023.csv')

    # dm = df.merge(dx,on='Document',how='inner').fillna('NAP')
    df['Gender'] = df['Gender'].str.replace('unknown','NA')
    dm = df[df['Topic']>=0].groupby(['CustomName','Country Code','Representation','Gender']).agg({'Document':'count'}).reset_index()
    fign = px.treemap(dm, path=[px.Constant('ECTRIMS'), 'CustomName','Country Code','Gender'], values='Document',hover_data=['Representation'])
    

    # Pre-calculate embeddings
    # embedding_model = SentenceTransformer("BAAI/bge-base-en")
    # embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
    
    
    # ## Cluster
    # umap_model = UMAP(n_neighbors=13, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    # hdbscan_model = HDBSCAN(min_cluster_size=5,min_samples=8, gen_min_span_tree=True, prediction_data=True)

    # # Pre-reduce embeddings for visualization purposes
    # reduced_embeddings = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)


    
    # umap_model = UMAP(n_neighbors=7, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    # hdbscan_model = HDBSCAN(min_cluster_size=7,min_samples=7, gen_min_span_tree=True, prediction_data=True)
    
    # # Pre-reduce embeddings for visualization purposes
    # reduced_embeddings = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)

    loaded_model = BERTopic.load("./test_conf_ectrims_2023_18_oct")
    #loaded_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings, hide_annotations=True, hide_document_hover=False, custom_labels=True)
    #st.markdown("<h4 style='text-align: center; color: black;'> Concept Analysis in Using Visulaization </h4>", unsafe_allow_html=True)
    fig = loaded_model.visualize_documents(titles, reduced_embeddings=embeddings, hide_annotations=True, hide_document_hover=False, custom_labels=True)
    #st.plotly_chart(fig, theme=None, use_container_width=True)
    #st.markdown("<h4 style='text-align: center; color: black;'> Concept View in Hierarchical Distribution </h4>", unsafe_allow_html=True)
    hierarchical_topics = loaded_model.hierarchical_topics(abstracts)
    fig1 = loaded_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics,custom_labels=True)
    #st.plotly_chart(fig1, theme=None, use_container_width=True)
    #st.markdown("<h4 style='text-align: center; color: black;'> Progressive View in Hierarchical Distribution </h4>", unsafe_allow_html=True)
    #fig2 = loaded_model.visualize_hierarchical_documents(abstracts, hierarchical_topics, reduced_embeddings=reduced_embeddings,custom_labels=True)
    #st.plotly_chart(fig2, theme=None, use_container_width=True)

    #df['Date']= pd.to_datetime(df['Date']).dt.date
    timestamps = df['Date'].to_list()
    topics_over_time = loaded_model.topics_over_time(docs=abstracts, 
                                    timestamps=timestamps, 
                                    global_tuning=True, 
                                    evolution_tuning=True, 
                                    nr_bins=30)
    #timestamps = list(pd.to_datetime(dx['Date']).dt.date)
    figt = loaded_model.visualize_topics_over_time(topics_over_time, top_n_topics=18,custom_labels=True)
    dc = pd.read_csv('./test_conf_ectrims_17_oct_2023/top_Summary.csv')
    drg = pd.read_csv('./test_conf_ectrims_2023_18_oct/rag_outcome.csv')
    tab6, tab5, tab0, tab1, tab2 = st.tabs(["Overall Summary","Temporal-View","Theme Deep Dive","Theme View in Spacial Distribution", "Theme View in Hierarchical Distribution"])
    with tab6:
        # Use the Streamlit theme.
        # This is the default. So you can also omit the theme argument.
        #col1, col2 = st.columns([1,1])
        st.markdown("<h4 style='text-align: center; color: black;'> Some Interesting Facts about Ectrims 2023 </h4>", unsafe_allow_html=True)
        cola, colb, colc, cold, cole, colf, colg = st.columns(7)
        cola.metric(label="Number of Unique Post", value=df.shape[0])
        colb.metric(label="Number of Authors", value=df['Author'].nunique())
        colc.metric(label="Number of Interest Listed", value=df['Interest'].nunique())
        cold.metric(label="Number of Unique Profession ", value=df['Professions'].nunique())
        cole.metric(label="Max Retweeted", value=df['Twitter Retweets'].max())
        colf.metric(label="Number of Countries", value=df['Country Code'].nunique())
        colg.metric(label="Number of Categories", value=df['Category Details'].nunique())
        #colc.metric(label="Number of Font Size Used", value=tt.size*1000.nunique())

        st.markdown("<h4 style='text-align: center; color: black;'> Examine Target Question and Explore Insights </h4>", unsafe_allow_html=True)
        rag = st.selectbox("Select Question of Interest",(drg['Question'].to_list()))
        st.markdown("<h6 style='text-align: center; color: black;'> Generated Response </h6>", unsafe_allow_html=True)
        st.markdown(drg[drg['Question']==rag]['Response'].reset_index(drop=True)[0])
        st.markdown("<h6 style='text-align: center; color: black;'> Referred Source Documents before Compilation of Response Generation </h6>", unsafe_allow_html=True)
        with stylable_container(
            "codeblock",
            """
            code {
                white-space: pre-wrap !important;
            }
            """,
        ):
            st.code(drg[drg['Question']==rag]['Source_Data'].reset_index(drop=True)[0])
                           
        
#         st.markdown("<h4 style='text-align: center; color: black;'> Overall Summary </h4>", unsafe_allow_html=True)
#         st.write("""Multiple sclerosis (MS) is a complex and debilitating autoimmune disease that affects millions of people worldwide. Recent research presented at the MS Milan conference has shed light on various aspects of the disease, including its pathophysiology, diagnosis, treatment options, and management strategies. Here are some key takeaways from the conference:

# 1. Longitudinal stability of autoantibody profiles: A study presented at the conference identified a subgroup of individuals with MS who exhibited a stable autoantibody profile throughout the course of their disease, both before and after symptom onset. This finding suggests that these autoantibodies may play a role in the development and progression of MS.
# 2. Thalamic atrophy: Another study discussed the role of the thalamus as a relay station in MS and the significance of thalamic atrophy in the disease process. Thalamic atrophy refers to the degeneration of the thalamus, a part of the brain responsible for sensory processing and transmission.
# 3. Genetic loci for MS susceptibility and severity: Professor Stephen Hauser's opening lecture at the conference focused on patterns of autoimmunity in different conditions and their potential implications for MS. He also discussed the genetic loci for MS susceptibility and severity, which have important implications for understanding the underlying mechanisms of the disease.
# 4. Advancements in molecular medicine for MS: The same lecture highlighted the significant progress made in molecular medicine for MS, particularly in the areas of diagnostics and treatments. New technologies such as next-generation sequencing and advanced imaging techniques have improved our ability to detect and monitor MS.
# 5. Evolution of treatment strategies: Ofatumumab, a drug used to treat MS, has been shown to hold the line in patients previously treated with anti-CD20 therapies. Meanwhile, biosimilar natalizumab has demonstrated promising results in treating MS, offering a potentially safer and more cost-effective alternative to existing treatments.
# 6. Breastfeeding recommendations: The conference addressed the controversial issue of breastfeeding in MS patients, with experts recommending caution when advising women about breastfeeding while taking disease-modifying therapies (DMTs).
# 7. Family planning in women of childbearing age with MS: Another hot topic at the conference was family planning in women of childbearing age with MS. Experts stressed the need for careful consideration and individualized advice regarding contraception and pregnancy planning in these patients.
# 8. Safety of anti-CD20 monoclonal antibodies in MS pregnancies: The conference also discussed the safety of anti-CD20 monoclonal antibodies in MS pregnancies. While previous studies raised concerns about the risk of adverse outcomes, newer data suggest that these therapies may be safe to use during pregnancy.
# 9. Importance of global health in MS research: The conference emphasized the importance of global health in MS research and treatment. With MS affecting people worldwide, there is a growing need for collaboration and knowledge sharing across countries and cultures.
# 10. Commitment to advancing MS research and patient care: Finally, the conference reaffirmed the commitment of the scientific community to advancing MS research and patient care. By bridging science and clinical practice, researchers aim to develop more effective treatments and improve health outcomes for those affected by MS."""
# )
        # #col1, col2 = st.columns([1,1])
        # st.markdown("<h4 style='text-align: center; color: black;'> Some Interesting Facts about Ectrims 2023 </h4>", unsafe_allow_html=True)
        # colh, coli, colj, colk, coll, colf, colg = st.columns(7)
        # author = df[['Author','Twitter Followers']].drop_duplicates().sort_values('Twitter Followers',ascending=False)['Author'][:5]
        # colh.write(author)

        
    
    with tab5:
        # Use the Streamlit theme.
        # This is the default. So you can also omit the theme argument.
        st.plotly_chart(figt, theme="streamlit", use_container_width=True)
    
    with tab0:
        # Use the Streamlit theme.
        # This is the default. So you can also omit the theme argument.
        st.plotly_chart(fign, theme="streamlit", use_container_width=True)
    
    with tab1:
        # Use the Streamlit theme.
        # This is the default. So you can also omit the theme argument.
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        # Use the native Plotly theme.
        st.plotly_chart(fig1, theme=None, use_container_width=True)

    # with tab3:
    #     # Use the native Plotly theme.
    #     st.plotly_chart(fig2, theme=None, use_container_width=True)







page_names_to_funcs = {
    "Concept Analysis": concept_view_1,
    "Summary Analysis": main_page,
}

selected_page = st.sidebar.selectbox("# Analysis Selection", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
