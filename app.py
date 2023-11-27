import streamlit as st
from jobClassification.pipeline.prediction import PredictionPipeline

st.title('Welcome to Job O*NET Classification App')
st.title('🔎 _Enter_ Job Details!')
st.markdown("""Enter Both Title and details""")

with st.form("search_form"):
	job_title = st.text_input("Enter Job Title:", placeholder = "job title")
    job_body = st.text_input("Enter Job Details:", placeholder = "job details")
    top_k = st.text_input("Enter options to display (Default 1):", placeholder = "How many O*NETs?")
    
	submit_status = st.form_submit_button("Search")
    
	if submit_status:

        if top_k !="":
            top_k = int(top_k)
        else:
            top_k = 1

		pred_obj = PredictionPipeline()

		output_df = pred_obj.predict(job_title,job_body, top_k)
		st.dataframe(output_df,hide_index=True)
      