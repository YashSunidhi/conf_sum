import streamlit as st
import pandas as pd
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

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: black;'> Conference Assesment Tool </h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: black;'> Understanding Discussion, Observation and Opinions </h6>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload a Summary csv File",type= 'csv' , key="file")
uploaded_file1 = st.sidebar.file_uploader("Upload a Concept csv File",type= 'csv' , key="file1")
uploaded_file2 = st.sidebar.file_uploader("Upload a Summary NR csv File",type= 'csv' , key="file2")
if uploaded_file is not None:
    if uploaded_file1 is not None:
        if uploaded_file2 is not None:
            def main_page(uploaded_file = uploaded_file,uploaded_file1 = uploaded_file1,uploaded_file2 = uploaded_file2):
                #create_space(1)
                with st.expander(label="# To Compare different Summarization and Provide feedback on what good will look like", expanded=False):
                    st.markdown("""

                    1) Upload Summary of Representative Document

                    2) Upload Document Based Concept Modelled File 

                    3) Upload Summary of Non- Representative Document

                    4) Observe and Provide Comment
                    """)
                "---"

                st.markdown("<h3 style='text-align: center; color: grey;'> Summary Comparision Tool </h3>", unsafe_allow_html=True)
                col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
                
                df = pd.read_csv(uploaded_file)
                df['User_Input'] = ''
                df1 = pd.read_csv(uploaded_file1)
                df2 = pd.read_csv(uploaded_file2)
                concept_option = st.sidebar.selectbox(
                'Concept Selection',
                (df['Topics'].unique()))
                
                collector = FeedbackCollector(
                    email='smnitrkl50@gmail.com',
                    password='Ram@2107',
                    project="default"
                )
            

                #### Type 1
                col1.markdown("<h5 style='text-align: center; color: grey;'> Representative Docs base Summary (Problem-Solution Structure) </h5>", unsafe_allow_html=True)
                do1 = df[(df['Topics']==concept_option) & (df['Summary Type']=='Problem-Solution Structure')]['Summary Variants'].reset_index(drop=True)[0]
                col1.write(do1)

                messages = str(do1)
                feedback_key = True

                for n, msg in enumerate(messages):
                    if feedback_key not in st.session_state:
                        st.session_state[feedback_key] = None
                    feedback = collector.st_feedback(
                        component="default",
                        feedback_type="thumbs",
                        open_feedback_label="[Optional] Provide additional feedback",
                        model='llama-13b',
                        key=feedback_key,
                        prompt_id=st.session_state.prompt_ids[int(n / 2) - 1],
                    )
                    if feedback:
                        with st.sidebar:
                            st.write(":orange[Here's the raw feedback you sent to [Trubrics](https://trubrics.streamlit.app/):]")
                            st.write(feedback)

                with col1:
                    exapnder = st.expander("Document Used")
                    do11 = df1[(df1['CustomName']==concept_option) & (df1['Representative_document']==True)]['Document'].reset_index(drop=True)
                    exapnder.write(pd.DataFrame(do11).to_html(escape=False), unsafe_allow_html=True)

                
                col1.markdown("<h5 style='text-align: center; color: grey;'> Non-Representative Docs base Summary(Problem-Solution Structure) </h5>", unsafe_allow_html=True)
                do111 = df2[(df2['Topics']==concept_option) & (df2['Summary Type']=='Problem-Solution Structure')]['Summary Variants'].reset_index(drop=True)[0]
                col1.write(do111)

                with col1:
                    exapnder = st.expander("Document Used")
                    do11 = df1[(df1['CustomName']==concept_option) & (df1['Representative_document']==False)]['Document'].reset_index(drop=True)
                    exapnder.write(pd.DataFrame(do11).to_html(escape=False), unsafe_allow_html=True)


                #### Type 2
                col2.markdown("<h5 style='text-align: center; color: grey;'> Representative Docs base Summary (Hierarchy and Structure) </h5>", unsafe_allow_html=True)
                do1 = df[(df['Topics']==concept_option) & (df['Summary Type']=='Hierarchy and Structure')]['Summary Variants'].reset_index(drop=True)[0]
                col2.write(do1)

                with col2:
                    exapnder = st.expander("Document Used")
                    do11 = df1[(df1['CustomName']==concept_option) & (df1['Representative_document']==True)]['Document'].reset_index(drop=True)
                    exapnder.write(pd.DataFrame(do11).to_html(escape=False), unsafe_allow_html=True)

                
                col2.markdown("<h5 style='text-align: center; color: grey;'> Non-Representative Docs base Summary(Hierarchy and Structure) </h5>", unsafe_allow_html=True)
                do111 = df2[(df2['Topics']==concept_option) & (df2['Summary Type']=='Hierarchy and Structure')]['Summary Variants'].reset_index(drop=True)[0]
                col2.write(do111)

                with col2:
                    exapnder = st.expander("Document Used")
                    do11 = df1[(df1['CustomName']==concept_option) & (df1['Representative_document']==False)]['Document'].reset_index(drop=True)
                    exapnder.write(pd.DataFrame(do11).to_html(escape=False), unsafe_allow_html=True)


                #### Type 3
                col3.markdown("<h5 style='text-align: center; color: grey;'> Representative Docs base Summary (Logical Flow of Arguments) </h5>", unsafe_allow_html=True)
                do1 = df[(df['Topics']==concept_option) & (df['Summary Type']=='Logical Flow of Arguments')]['Summary Variants'].reset_index(drop=True)[0]
                col3.write(do1)

                with col3:
                    exapnder = st.expander("Document Used")
                    do11 = df1[(df1['CustomName']==concept_option) & (df1['Representative_document']==True)]['Document'].reset_index(drop=True)
                    exapnder.write(pd.DataFrame(do11).to_html(escape=False), unsafe_allow_html=True)

                
                col3.markdown("<h5 style='text-align: center; color: grey;'> Non-Representative Docs base Summary(Logical Flow of Arguments) </h5>", unsafe_allow_html=True)
                do111 = df2[(df2['Topics']==concept_option) & (df2['Summary Type']=='Logical Flow of Arguments')]['Summary Variants'].reset_index(drop=True)[0]
                col3.write(do111)

                with col3:
                    exapnder = st.expander("Document Used")
                    do11 = df1[(df1['CustomName']==concept_option) & (df1['Representative_document']==False)]['Document'].reset_index(drop=True)
                    exapnder.write(pd.DataFrame(do11).to_html(escape=False), unsafe_allow_html=True)


                #### Type 4
                col4.markdown("<h5 style='text-align: center; color: grey;'> Representative Docs base Summary (Retrospectives and Prospectives) </h5>", unsafe_allow_html=True)
                do1 = df[(df['Topics']==concept_option) & (df['Summary Type']=='Retrospectives and Prospectives')]['Summary Variants'].reset_index(drop=True)[0]
                col4.write(do1)

                with col4:
                    exapnder = st.expander("Document Used")
                    do11 = df1[(df1['CustomName']==concept_option) & (df1['Representative_document']==True)]['Document'].reset_index(drop=True)
                    exapnder.write(pd.DataFrame(do11).to_html(escape=False), unsafe_allow_html=True)

                
                col4.markdown("<h5 style='text-align: center; color: grey;'> Non-Representative Docs base Summary(Retrospectives and Prospectivese) </h5>", unsafe_allow_html=True)
                do111 = df2[(df2['Topics']==concept_option) & (df2['Summary Type']=='Retrospectives and Prospectives')]['Summary Variants'].reset_index(drop=True)[0]
                col4.write(do111)

                with col4:
                    exapnder = st.expander("Document Used")
                    do11 = df1[(df1['CustomName']==concept_option) & (df1['Representative_document']==False)]['Document'].reset_index(drop=True)
                    exapnder.write(pd.DataFrame(do11).to_html(escape=False), unsafe_allow_html=True)


                #### Type 5
                col5.markdown("<h5 style='text-align: center; color: grey;'> Representative Docs base Summary (Correlations and Associations) </h5>", unsafe_allow_html=True)
                do1 = df[(df['Topics']==concept_option) & (df['Summary Type']=='Correlations and Associations')]['Summary Variants'].reset_index(drop=True)[0]
                col5.write(do1)

                with col5:
                    exapnder = st.expander("Document Used")
                    do11 = df1[(df1['CustomName']==concept_option) & (df1['Representative_document']==True)]['Document'].reset_index(drop=True)
                    exapnder.write(pd.DataFrame(do11).to_html(escape=False), unsafe_allow_html=True)

                
                col5.markdown("<h5 style='text-align: center; color: grey;'> Non-Representative Docs base Summary(Correlations and Associations) </h5>", unsafe_allow_html=True)
                do111 = df2[(df2['Topics']==concept_option) & (df2['Summary Type']=='Correlations and Associations')]['Summary Variants'].reset_index(drop=True)[0]
                col5.write(do111)

                with col5:
                    exapnder = st.expander("Document Used")
                    do11 = df1[(df1['CustomName']==concept_option) & (df1['Representative_document']==False)]['Document'].reset_index(drop=True)
                    exapnder.write(pd.DataFrame(do11).to_html(escape=False), unsafe_allow_html=True)

            def concept_view_1():
               
                df = pd.read_csv('./test_conf_summ/final_doc_input.csv')
                abstracts = df['Full Text'].to_list()
                titles = df["Title"].to_list()

                # Pre-calculate embeddings
                embedding_model = SentenceTransformer("BAAI/bge-base-en")
                embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
                
                
                ## Cluster
                
                umap_model = UMAP(n_neighbors=7, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
                hdbscan_model = HDBSCAN(min_cluster_size=7,min_samples=7, gen_min_span_tree=True, prediction_data=True)
                
                # Pre-reduce embeddings for visualization purposes
                reduced_embeddings = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)

                loaded_model = BERTopic.load("./test_conf_summ")
                #loaded_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings, hide_annotations=True, hide_document_hover=False, custom_labels=True)
                st.markdown("<h4 style='text-align: center; color: black;'> Concept Analysis in Using Visulaization </h4>", unsafe_allow_html=True)
                fig = loaded_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings, hide_annotations=True, hide_document_hover=False, custom_labels=True)
                #st.plotly_chart(fig, theme=None, use_container_width=True)
                #st.markdown("<h4 style='text-align: center; color: black;'> Concept View in Hierarchical Distribution </h4>", unsafe_allow_html=True)
                hierarchical_topics = loaded_model.hierarchical_topics(abstracts)
                fig1 = loaded_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics,custom_labels=True)
                #st.plotly_chart(fig1, theme=None, use_container_width=True)
                #st.markdown("<h4 style='text-align: center; color: black;'> Progressive View in Hierarchical Distribution </h4>", unsafe_allow_html=True)
                fig2 = loaded_model.visualize_hierarchical_documents(abstracts, hierarchical_topics, reduced_embeddings=reduced_embeddings,custom_labels=True)
                #st.plotly_chart(fig2, theme=None, use_container_width=True)

                tab1, tab2, tab3 = st.tabs(["Concept View in Spacial Distribution", "Concept View in Hierarchical Distribution","Progressive View in Hierarchical Distribution"])
                with tab1:
                    # Use the Streamlit theme.
                    # This is the default. So you can also omit the theme argument.
                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                with tab2:
                    # Use the native Plotly theme.
                    st.plotly_chart(fig, theme=None, use_container_width=True)

                with tab3:
                    # Use the native Plotly theme.
                    st.plotly_chart(fig, theme=None, use_container_width=True)







    page_names_to_funcs = {
        #"Concept Analysis": concept_view_1,
        "Summary Analysis": main_page,
    }

    selected_page = st.sidebar.selectbox("# Analysis Selection", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()
