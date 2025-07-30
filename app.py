import streamlit as st

# Set page config
st.set_page_config(page_title="Yuono Dwi Raharjo - Portfolio", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About Me", "This Project"])

# Main content
if page == "About Me":
    st.title("👨‍💻 About Me")
    
    # Profile header
    st.markdown("""
    ### Yuono Dwi Raharjo
    **Data Science and Assessment Enthusiast**  
    📍 Yogyakarta, Indonesia | 📧 yuonodwiraharjo26@gmail.com  
    🔗 [LinkedIn](https://linkedin.com/in/yuonodraharjo) | 📱 +62 812-2706-8269
    """)
    
    # Professional Summary
    st.markdown("""
    ## Professional Summary
    Passionate Data and Research professional with an academic background in Psychology and extensive experience in psychometrics, 
    data analysis, and research development. Combines technical expertise with psychological insights to effectively 
    support data-driven decision-making in data science, psychometrics, and business.
    """)
    
    # Skills
    st.markdown("## 🛠 Technical Skills")
    
    skills_col1, skills_col2, skills_col3 = st.columns(3)
    
    with skills_col1:
        st.markdown("""
        **Data Analysis & Visualization**  
        • Python (Pandas, NumPy, SciPy)  
        • SQL (BigQuery, PostgreSQL)  
        • R, Jamovi, SPSS  
        • Power BI, Tableau  
        """)
    
    with skills_col2:
        st.markdown("""
        **Psychological Assessment**  
        • WAIS, CFIT, BDI, BAI  
        • 16PF, Projective Tests  
        • Test Development  
        • Psychometric Analysis  
        """)
    
    with skills_col3:
        st.markdown("""
        **Research & Analytics**  
        • Quantitative Research  
        • Market Research  
        • People Analytics  
        • SEM, Factor Analysis  
        """)
    
    # Education
    st.markdown("## 🎓 Education")
    
    edu_col1, edu_col2 = st.columns(2)
    
    with edu_col1:
        st.markdown("""
        **Bootcamp Data Science**  
        *Dibimbing.id | Feb 2025-Present*  
        • Score: 98/100  
        • Python, SQL, EDA  
        • Web Scraping, Marketing Analytics  
        """)
        
    with edu_col2:
        st.markdown("""
        **Bachelors in Psychology**  
        *Universitas Gadjah Mada | 2021-2025*  
        • GPA: 3.85/4.00  
        • Statistics, Research Methods  
        • Psychological Measurement  
        """)
    
    # Work Experience
    st.markdown("## 💼 Professional Experience")
    
    with st.expander("**Development Staff** at UPAP Psikologi UGM (Jan-Dec 2024)", expanded=True):
        st.markdown("""
        • Developed **100+ assessment items** for academic potential and situational judgment tests  
        • Managed **test digitalization** for CAT-based assessments (4,000+ items bank)  
        • Conducted **psychometric analysis** using Jamovi, R, and Winstep  
        • Authored **10 assessment manuals** for Hak Kekayaan Intelektual registration  
        • Organized **10+ training sessions** in statistics and psychometrics  
        """)
    
    with st.expander("**Technical Staff Intern** at Psimetrika Indonesia (Jul-Nov 2023)"):
        st.markdown("""
        • Developed **12 cognitive test items** on digital ethics for government officials  
        • Performed **Classical Test Theory analysis** using R  
        • Conducted **Confirmatory Factor Analysis** for instrument validation  
        • Created **scoring reports** and analysis templates  
        """)

elif page == "This Project":
    st.title("🧠 About This Project")
    
    st.markdown("""
    ## Social Anxiety Profiling Dashboard
    
    This interactive dashboard demonstrates the intersection of psychological assessment and data science, 
    developed to provide actionable insights into anxiety patterns.
    """)
    
    st.markdown("""
    ### Project Objectives:
    • Analyze patterns in anxiety-related data using psychometric approaches  
    • Create interactive visualizations for data exploration  
    • Bridge psychological theory with data science applications  
    """)
    
    st.markdown("""
    ### Technical Implementation:
    • **Frontend**: Streamlit for interactive web interface  
    • **Data Processing**: Pandas for efficient data manipulation  
    • **Visualization**: Altair for dynamic, interactive charts  
    • **Analysis**: Incorporates psychometric validation techniques  
    """)
    
    st.markdown("""
    ### Professional Value:
    This project showcases my unique combination of:
    - Psychological assessment expertise
    - Data science technical skills
    - Ability to create practical analytical tools
    """)
    
    st.markdown("""
    ### Connect for Collaboration
    Interested in this project or similar applications?  
    Contact me at: yuonodwiraharjo26@gmail.com
    """)

