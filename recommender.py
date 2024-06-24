"""
Recommendation System

Links
	- https://towardsdatascience.com/how-to-build-a-recommendation-engine-quick-and-simple-aec8c71a823e
	- https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/


source_path = os.path.join("data/coursera-courses.csv")
"""

import streamlit as st
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt 
import altair as alt

from rake_nltk import Rake
# from nltk.corpus import stopwords 
import nltk
nltk.download('stopwords')
nltk.download('punkt')

### pip install scikit-learn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

FILTERED_COURSES = None
SELECTED_COURSE = None

# Phương thức mà xử lý và "làm sạch" tên dữ liệu
@st.cache(persist=True)
def clean_col_names(df, columns):
	"""
	Cleans column names
	-----
	columns:
		List of column names
	"""

	new = []
	for c in columns:
		new.append(c.lower().replace(' ','_'))
	return new

# Phương thức để chuẩn bị dữ liệu bằng cách làm sạch tên cột, 
# diễn đạt and thay thế những giá trị bị thiếu
@st.cache(persist=True)
def prepare_data(df):
	"""
	Prepares the final dataset
	-----
	df:
		dataframe
	"""

	# clean column names
	df.columns = clean_col_names(df, df.columns)

	# impute missing values that creeped in
	df['skills'] = df['skills'].fillna('Missing')
	df['instructors'] = df['instructors'].fillna('Missing')

	# making certain features numeric
	def make_numeric(x):
		if(x=='Missing'):
			return np.nan
		return float(x)

	df['course_rating'] = df['course_rating'].apply(make_numeric)
	df['course_rated_by'] = df['course_rated_by'].apply(make_numeric)
	df['percentage_of_new_career_starts'] = df['percentage_of_new_career_starts'].apply(make_numeric)
	df['percentage_of_pay_increase_or_promotion'] = df['percentage_of_pay_increase_or_promotion'].apply(make_numeric)

	def make_count_numeric(x):
	    if('k' in x):
	        return (float(x.replace('k','')) * 1000)
	    elif('m' in x):
	        return (float(x.replace('m','')) * 1000000)
	    elif('Missing' in x):
	        return (np.nan)

	df['enrolled_student_count'] = df['enrolled_student_count'].apply(make_count_numeric)

    # extract time to complete
	def find_time(x):
	    l = x.split(' ')
	    idx = 0
	    for i in range(len(l)):
	        if(l[i].isdigit()):
	            idx = i 
	    try:
	        return (l[idx] + ' ' + l[idx+1])
	    except:
	        return l[idx]

	df['estimated_time_to_complete'] = df['estimated_time_to_complete'].apply(find_time)

	# split by skills
	def split_it(x):
		return (x.split(','))
	df['skills'] = df['skills'].apply(split_it)

	return df

# Phương thức tải dữ liệu và xử lý tiền chuẩn bị cho dữ liệu
@st.cache(allow_output_mutation=True)
def load_data():
	source_path1 = os.path.join("data/coursera-courses-overview.csv")
	source_path2 = os.path.join("data/coursera-individual-courses.csv")
	df_overview = pd.read_csv(source_path1)
	df_individual = pd.read_csv(source_path2)
	df = pd.concat([df_overview, df_individual], axis=1)

	# preprocess it now
	df = prepare_data(df)

	return df

# Phương thức bắt dữ liệu và chọn ra những khóa học được lựa chọn từ tiêu chí đã chọn
@st.cache(persist=True)
def filter(dataframe, chosen_options, feature, id):
	selected_records = []
	for i in range(1000):
		for op in chosen_options:
			if op in dataframe[feature][i]:
				selected_records.append(dataframe[id][i])
	return selected_records

# Phương thức loại ra từ khóa từ miêu tả khóa học
def extract_keywords(df, feature):
    r = Rake()
    keyword_lists = []
    for i in range(1000):
        descr = df[feature][i]
        r.extract_keywords_from_text(descr)
        key_words_dict_scores = r.get_word_degrees()
        keywords_string = " ".join(list(key_words_dict_scores.keys()))
        keyword_lists.append(keywords_string)
        
    return keyword_lists

def extract_keywords(df, feature):

    r = Rake()
    keyword_lists = []
    for i in range(df[feature].shape[0]):
        descr = df[feature][i]
        r.extract_keywords_from_text(descr)
        key_words_dict_scores = r.get_word_degrees()
        keywords_string = " ".join(list(key_words_dict_scores.keys()))
        keyword_lists.append(keywords_string)
        
    return keyword_lists

# Đưa ra những đề xuất những khóa học tương tự hoặc khác với khóa học được chọn
def recommendations(df, input_course, cosine_sim, find_similar=True, how_many=5):
    
    # Khởi tạo danh sách khóa học được đề xuất
    recommended = []

	# Tìm khóa học trong dataframe khớp với khóa học được nhập vào
    selected_course = df[df['course_name']==input_course]
    
    # index of the course fed as input (Lấy chỉ mục của khóa học nhập vào)
    idx = selected_course.index[0]


	# Tạo một series mới với các chỉ số tương tự sắp xếp từ cao đến thấp
    if(find_similar):	# Nếu muốn tìm khóa học tương tự
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    else:	# Nếu muốn tìm khóa học không tương tự
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = True)


    # getting the indexes of the top 'how_many' courses
	# (Lấy chỉ mục của top số lượng khóa học được chỉ định)
    if(len(score_series) < how_many):
    	how_many = len(score_series)
    top_sugg = list(score_series.iloc[1:how_many+1].index)
    
    # populating the list with the titles of the best 10 matching movies
	# (Thêm tiêu đề của các khóa học đề xuất vào danh sách recommended)
    for i in top_sugg:
        qualified = df['course_name'].iloc[i]
        recommended.append(qualified)
        
	# Trả về danh sách khóa học được đề xuất
    return recommended

# Đưa ra những đề xuất dựa trên nội dung của dữ liệu đã chọn
def content_based_recommendations(df, input_course, courses):

	# filter out the courses
	df = df[df['course_name'].isin(courses)].reset_index()

	# create description keywords
	df['descr_keywords'] = extract_keywords(df, 'description')

	# instantiating and generating the count matrix
	count = CountVectorizer()
	count_matrix = count.fit_transform(df['descr_keywords'])

	# generating the cosine similarity matrix
	cosine_sim = cosine_similarity(count_matrix, count_matrix)

	# make the recommendation
	rec_courses_similar = recommendations(df, input_course, cosine_sim, True)
	temp_sim = df[df['course_name'].isin(rec_courses_similar)]
	rec_courses_dissimilar = recommendations(df, input_course, cosine_sim, False)
	temp_dissim = df[df['course_name'].isin(rec_courses_dissimilar)]

	# top 3
	st.write("Top 5 most similar courses (table format)")
	st.write(temp_sim)
	st.write("Top 5 most dissimilar courses (table format)")
	st.write(temp_dissim)

# Tiền chuẩn bị cho đề xuất dựa trên nội dung
def prep_for_cbr(df):
    st.header("Content-based Recommendation")
    st.sidebar.header("Filter on Preferences")
    st.write("This section is entrusted with the responsibility of"
             " analysing a filtered subset of courses based on the **skills**"
             " a learner is looking to develop. This filter can be adjusted on"
             " the sidebar.")
    st.write("This section also finds courses similar to a selected course"
             " based on Content-based recommendation. The learner can choose"
             " any course that has been filtered on the basis of their skills"
             " in the previous section.")
    st.write("Choose course from 'Select Course' dropdown on the sidebar")

    # filter by skills
    skills_avail = []
    for i in range(1000):
        skills_avail = skills_avail + df['skills'][i]
    skills_avail = list(set(skills_avail))
    skills_select = st.sidebar.multiselect("Select Skills", skills_avail)

    temp = filter(df, skills_select, 'skills', 'course_url')
    skill_filtered = df[df['course_url'].isin(temp)].reset_index()

    st.write("### Filtered courses based on skill preferences")




    # Display data in Card format grid manner
	# Display data in Card format grid manner
    st.write("### Courses Data Table")

    st.write(
        '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px;">',
        unsafe_allow_html=True,
    )
    card_count = 0
    for index, row in skill_filtered.iterrows():
        # Shorten the description with an ellipsis if too long
        max_desc_length = 100
        desc = row['description'][:max_desc_length] + ("..." if len(row['description']) > max_desc_length else "")
        
        # Add star rating
        rating = int(row['course_rating'])
        stars = ''.join(['⭐' for _ in range(rating)])

        st.markdown(f"<div style='background-color:#e6e6fa;padding:20px;border-radius:5px; margin:10px'>"
                    f"<h4>{row['course_name']}</h4>"
                    f"<p>{desc}</p>"
                    f"<p>Rating: {stars}</p>"
                    f"<p><strong>Provided by:</strong> {row['course_provided_by']}</p>"
                    f"</div>", unsafe_allow_html=True)

        card_count += 1
        if card_count % 3 == 0:
            st.write('</div>', unsafe_allow_html=True)
            st.write('<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px;">', unsafe_allow_html=True)

    # Check if there are any remaining cards to close the grid
    if card_count % 3 != 0:
        st.write('</div>', unsafe_allow_html=True)




    # Display the data in a table format
    st.write("### Courses Data Table")
    st.write(skill_filtered)

    st.write("**Number of programmes filtered:**", skill_filtered.shape[0])
    st.write("**Number of courses:**",
              skill_filtered[skill_filtered['learning_product_type'] == 'COURSE'].shape[0])
    st.write("**Number of professional degrees:**",
              skill_filtered[skill_filtered['learning_product_type'] == 'PROFESSIONAL CERTIFICATE'].shape[0])
    st.write("**Number of specializations:**",
              skill_filtered[skill_filtered['learning_product_type'] == 'SPECIALIZATION'].shape[0])

    chart = alt.Chart(skill_filtered).mark_bar().encode(
        y='course_provided_by:N',
        x='count(course_provided_by):Q'
    ).properties(
        title='Organizations providing these courses'
    )
    st.altair_chart(chart)

    if len(skill_filtered)<=2:
        st.write("*There should be at least 3 courses. Do add more.*")

    input_course = st.sidebar.selectbox("Select Course", skill_filtered['course_name'].tolist(), key='courses')
    rec_radio = st.sidebar.radio("Recommend Similar Courses", ('no', 'yes'), index=0)
    if rec_radio == 'yes':
        content_based_recommendations(df, input_course, skill_filtered['course_name'].tolist())


# main to render web app
# def main():


# 	st.title("CouReco")
# 	st.write("Exploring Courses on Coursera")
# 	st.sidebar.title("Set your Parameters")
# 	st.sidebar.header("Preliminary Inspection")
# 	st.header("About the Project")
# 	st.write("CouReco is a minimalistic system built to help learners"
# 		" navigate through the courses on Coursera, aided by a"
# 		" data-driven strategy. A learner could visualize different"
# 		" features provided in the dataset or interact with this app"
# 		" to find suitable courses to take. CouReco also can help"
# 		" identify suitable courses for a learner based on their"
# 		" learning preferences.")

# 	# load and disp data
# 	df = load_data()
# 	st.header("Dataset Used")
# 	st.write("For the purpose of building CouReco, data from Coursera"
# 		" was scraped using the requests and beautifulsoup4 libraries."
# 		" The final dataset thus acquired consists of 1000 instances"
# 		" and 14 features.")
# 	st.markdown("Toggle the **Display raw data** checkbox on the sidebar"
# 		" to show or hide the dataset.")
# 	# toggle button to display raw data
# 	if st.sidebar.checkbox("Display raw data", key='disp_data'):
# 		st.write(df)
# 	else:
# 		pass
# 	st.markdown("### What does each feature represent?")
# 	st.write("**course_url:** URL to the course homepage")
# 	st.write("**course_name:** Name of the course")
# 	st.write("**learning_product_type:** Is it a course, a professional certificate or a specialization?")
# 	st.write("**course_provided_by:** Partner providing the course")
# 	st.write("**course_rating:** Overall rating of the course")
# 	st.write("**course_rated_by:** Number of learners who rated the course")
# 	st.write("**enrolled_student_count:** Number of learners enrolled")
# 	st.write("**course_difficulty:** Difficulty level of the course")
# 	st.write("**skills:** Relevant skills the course will deal with")
# 	st.write("**description:** About the course")
# 	st.write("**percentage_of_new_career_starts:** Number of learners who started a new career after taking this course")
# 	st.write("**percentage_of_pay_increase_or_promotion:** Number of learners who received a pay increase or promotion after taking this course")
# 	st.write("**estimated_time_to_complete:** Approximate time to complete")
# 	st.write("**instructors:** Instructors of the course")

# 	# initiate CBR
# 	prep_for_cbr(df)


def main():
    st.sidebar.title("Set your Parameters")
    st.sidebar.header("Preliminary Inspection")
    
    st.title("CouReco")
    
    with st.expander("About this Recommender System"):
        st.write(
            "CouReco is a minimalistic system built to help learners navigate through the courses on Coursera,"
            "aided by a data-driven strategy. A learner could visualize different features provided in the dataset or "
            "interact with this app to find suitable courses to take. CouReco also can help identify suitable "
            "courses for a learner based on their learning preferences.")
        st.markdown("### What does each feature represent?")
        st.write("**course_url:** URL to the course homepage")
        st.write("**course_name:** Name of the course")
        st.write("**learning_product_type:** Is it a course, a professional certificate or a specialization?")
        st.write("**course_provided_by:** Partner providing the course")
        st.write("**course_rating:** Overall rating of the course")
        st.write("**course_rated_by:** Number of learners who rated the course")
        st.write("**enrolled_student_count:** Number of learners enrolled")
        st.write("**course_difficulty:** Difficulty level of the course")
        st.write("**skills:** Relevant skills the course will deal with")
        st.write("**description:** About the course")
        st.write("**percentage_of_new_career_starts:** Number of learners who started a new career after taking this course")
        st.write("**percentage_of_pay_increase_or_promotion:** Number of learners who received a pay increase or promotion after taking this course")
        st.write("**estimated_time_to_complete:** Approximate time to complete")
        st.write("**instructors:** Instructors of the course")
        
    df = load_data()
    with st.expander("Raw Data Used"):
        st.write(df)

    with st.expander("Course Recommender System", expanded=True):
        st.header("Exploring Courses on Coursera")
        prep_for_cbr(df)

	
if __name__=="__main__":
	main()
