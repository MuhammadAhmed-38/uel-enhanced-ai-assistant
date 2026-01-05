if st.sidebar.button("🚪 Sign Out", key="sidebar_sign_out"):
            st.session_state.current_profile = None
            st.session_state.profile_active = False
            st.session_state.show_login = True
            st.rerun()
    else:
            st.sidebar.info("Please create or login to your profile")
            if st.sidebar.button("➕ Create Profile", key="sidebar_create_profile"):
                st.session_state.show_profile_creator = True
                st.session_state.show_login = False
            if st.sidebar.button("🔐 Login", key="sidebar_login"):
                st.session_state.show_login = True
                st.session_state.show_profile_creator = False


def render_profile_creator():
    """Render profile creation form with password and local saving."""
    st.header("👤 Create Student Profile")
    st.info("All fields marked with * are required.")
    
    with st.form("profile_creator_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            first_name = st.text_input("First Name *", key="new_first_name_input")
            last_name = st.text_input("Last Name *", key="new_last_name_input")
            email = st.text_input("Email *", key="new_email_input").lower()
            password = st.text_input("Password *", type="password", key="new_password_input")
            confirm_password = st.text_input("Confirm Password *", type="password", key="confirm_password_input")
            phone = st.text_input("Phone", key="new_phone_input")
            date_of_birth = st.date_input("Date of Birth", datetime(2000, 1, 1), key="new_dob_input")
        
        with col2:
            country = st.selectbox("Country *", 
                ["", "United Kingdom", "United States", "India", "China", "Nigeria", "Pakistan", "Canada", "Other"],
                key="new_country_input")
            nationality = st.selectbox("Nationality", 
                ["", "British", "American", "Indian", "Chinese", "Nigerian", "Pakistani", "Canadian", "Other"],
                key="new_nationality_input")
            city = st.text_input("City", key="new_city_input")
            postal_code = st.text_input("Postal Code", key="new_postal_input")
        
        st.subheader("📚 Academic Information")
        
        col3, col4 = st.columns(2)
        with col3:
            academic_level = st.selectbox("Current Academic Level *",
                ["", "high_school", "undergraduate", "graduate", "postgraduate", "masters", "phd"],
                key="new_academic_level_input")
            field_of_interest = st.selectbox("Field of Interest *",
                ["", "Computer Science", "Business Management", "Engineering", "Data Science", 
                 "Psychology", "Medicine", "Law", "Arts", "Other"],
                key="new_field_input")
            current_institution = st.text_input("Current Institution", key="new_institution_input")
        
        with col4:
            gpa = st.number_input("GPA (out of 4.0)", 0.0, 4.0, 3.0, 0.1, key="new_gpa_input")
            ielts_score = st.number_input("IELTS Score", 0.0, 9.0, 6.5, 0.5, key="new_ielts_input")
            graduation_year = st.number_input("Expected Graduation Year", 2020, 2030, 2024, key="new_grad_year_input")
        
        st.subheader("💼 Professional Background")
        work_experience = st.number_input("Years of Work Experience", 0, 20, 0, key="new_work_exp_input")
        job_title = st.text_input("Current Job Title", key="new_job_title_input")
        
        st.subheader("🎯 Preferences")
        career_goals = st.text_area("Career Goals", key="new_career_goals_input")
        interests = st.multiselect("Interests",
            ["Technology", "Business", "Research", "Healthcare", "Education", "Arts", "Sports"],
            key="new_interests_input")
        preferred_modules = st.text_input("Preferred Modules (comma-separated)", key="new_preferred_modules_input")
        
        submitted = st.form_submit_button("✅ Create Profile")
        
        if submitted:
            # Basic validation
            if not all([first_name, last_name, email, password, confirm_password, academic_level, field_of_interest, country]):
                st.error("❌ Please fill in all required fields marked with *")
            elif password != confirm_password:
                st.error("❌ Passwords do not match.")
            elif len(password) < 6:
                st.error("❌ Password must be at least 6 characters long.")
            else:
                try:
                    profile_data = {
                        'first_name': first_name,
                        'last_name': last_name,
                        'email': email,
                        'phone': phone,
                        'date_of_birth': str(date_of_birth),
                        'country': country,
                        'nationality': nationality,
                        'city': city,
                        'postal_code': postal_code,
                        'academic_level': academic_level,
                        'field_of_interest': field_of_interest,
                        'current_institution': current_institution,
                        'gpa': gpa,
                        'ielts_score': ielts_score,
                        'graduation_year': graduation_year,
                        'work_experience_years': work_experience,
                        'current_job_title': job_title,
                        'career_goals': career_goals,
                        'interests': interests,
                        'preferred_modules': [m.strip() for m in preferred_modules.split(',')] if preferred_modules else []
                    }
                    
                    profile = st.session_state.ai_system.profile_manager.create_profile(profile_data, password)
                    
                    st.success(f"🎉 Profile created successfully! Welcome {first_name}!")
                    st.balloons()
                    time.sleep(2)
                    st.session_state.show_profile_creator = False
                    st.rerun()
                    
                except ValueError as ve:
                    st.error(f"❌ Creation Error: {ve}")
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred: {e}")


def render_login_form():
    """Render the student login form."""
    st.header("🔐 Student Login")
    
    with st.form("login_form"):
        email = st.text_input("Email", key="login_email_input").lower()
        password = st.text_input("Password", type="password", key="login_password_input")
        
        login_button = st.form_submit_button("🔐 Login")

        if login_button:
            if not email or not password:
                st.error("Please enter both email and password.")
            else:
                with st.spinner("Authenticating..."):
                    profile_manager = st.session_state.ai_system.profile_manager
                    logged_in_profile = profile_manager.login_profile(email, password)
                    
                    if logged_in_profile:
                        st.success(f"Welcome back, {logged_in_profile.first_name}!")
                        st.session_state.show_login = False
                        st.session_state.profile_active = True
                        st.session_state.current_profile = logged_in_profile
                        st.rerun()
                    else:
                        st.error("Invalid email or password. Please try again or create a new profile.")


def render_main_interface():
    """Render main application interface"""
    if not st.session_state.get('system_ready', False):
        st.error("❌ System not ready. Please refresh the page.")
        return
    
    # Header
    st.title("🎓 University of East London - AI Assistant")
    st.markdown("*Your intelligent companion for university applications and student services*")
    
    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 AI Chat", "🎯 Course Recommendations", "📊 Admission Prediction", 
        "📄 Document Verification", "📈 Analytics"
    ])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        render_course_recommendations()
    
    with tab3:
        render_admission_prediction()
    
    with tab4:
        render_document_verification()
    
    with tab5:
        render_analytics_dashboard()


def render_chat_interface():
    """Render AI chat interface"""
    st.header("💬 AI Chat Assistant")
    
    # Voice input section
    ai_system = st.session_state.ai_system
    if ai_system.voice_service and ai_system.voice_service.is_available():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("🎤 Voice input available! Click the button to speak your question.")
        with col2:
            if st.button("🎤 Voice Input", key="voice_input_btn"):
                with st.spinner("🎧 Listening..."):
                    voice_text = ai_system.voice_service.speech_to_text()
                    if voice_text and not voice_text.startswith("❌"):
                        st.session_state.voice_input = voice_text
                        st.success(f"Heard: {voice_text}")
    else:
        st.warning("Voice service not available. Please check system status.")
    
    # Chat input
    user_input = st.text_input(
        "Ask me anything about UEL courses, applications, or university services:",
        value=st.session_state.get('voice_input', ''),
        key="chat_input"
    )
    
    # Clear voice input after use
    if 'voice_input' in st.session_state:
        del st.session_state.voice_input
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        send_clicked = st.button("📤 Send", key="send_btn")
    with col2:
        if ai_system.voice_service and ai_system.voice_service.is_available():
            if st.button("🔊 Speak Response", key="speak_btn"):
                if st.session_state.chat_history:
                    last_response = st.session_state.chat_history[-1].get('ai_response', '')
                    if last_response:
                        ai_system.voice_service.text_to_speech(last_response)
                        st.success("🔊 Speaking response...")
        else:
            st.button("🔊 Speak Response (Unavailable)", disabled=True, key="speak_btn_disabled")
    
    # Process message
    if send_clicked and user_input.strip():
        with st.spinner("🤖 Processing your message..."):
            current_profile = st.session_state.current_profile
            response_data = ai_system.process_user_message(user_input, current_profile)
            
            # Add to chat history
            chat_entry = {
                "user_message": user_input,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                **response_data
            }
            st.session_state.chat_history.append(chat_entry)
        
        # Clear input
        st.session_state.chat_input = ""
        st.rerun()
    
    # Display chat history
    st.markdown("---")
    
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
            st.markdown(f"**🙋 You ({chat['timestamp']}):**")
            st.markdown(chat['user_message'])
            
            st.markdown("**🤖 UEL AI Assistant:**")
            st.markdown(chat['ai_response'])
            
            # Show sentiment if available
            if 'sentiment' in chat:
                sentiment = chat['sentiment']
                if sentiment.get('emotions'):
                    st.caption(f"😊 Detected emotions: {', '.join(sentiment['emotions'])}")
            
            st.markdown("---")
    else:
        st.info("👋 Start a conversation! Ask me about courses, applications, or any UEL services.")


def render_course_recommendations():
    """Render course recommendation interface"""
    st.header("🎯 Personalized Course Recommendations")
    
    if not st.session_state.profile_active:
        st.warning("👤 Please create a profile to get personalized recommendations.")
        return
    
    current_profile = st.session_state.current_profile
    
    # Additional preferences
    with st.expander("🔧 Customize Recommendations"):
        col1, col2 = st.columns(2)
        with col1:
            preferred_level = st.selectbox("Preferred Level", 
                ["Any", "high_school", "undergraduate", "graduate", "postgraduate", "masters", "phd"], key="pref_level")
            study_mode = st.selectbox("Study Mode",
                ["Any", "full-time", "part-time", "online"], key="pref_mode")
        with col2:
            budget_max = st.number_input("Max Budget (£)", 0, 50000, 20000, key="pref_budget")
            start_date = st.selectbox("Preferred Start",
                ["Any", "September 2024", "January 2025"], key="pref_start")
    
    if st.button("🎯 Get Recommendations", key="get_recs_btn"):
        with st.spinner("🔍 Analyzing your profile and finding perfect matches..."):
            try:
                preferences = {
                    'level': preferred_level if preferred_level != "Any" else None,
                    'study_mode': study_mode if study_mode != "Any" else None,
                    'budget_max': budget_max,
                    'start_date': start_date if start_date != "Any" else None
                }
                
                recommendations = st.session_state.ai_system.course_recommender.recommend_courses(
                    current_profile.to_dict(), preferences
                )
                
                if recommendations:
                    st.success(f"🎉 Found {len(recommendations)} excellent matches for you!")
                    
                    for i, course in enumerate(recommendations):
                        with st.container():
                            # Course header with match quality
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.subheader(f"🎓 {course['course_name']}")
                                st.caption(f"🏢 {course['department']} • ⏱️ {course['duration']} • 📊 {course['level']}")
                            with col2:
                                st.markdown(f"**{course['match_quality']}**")
                                st.progress(course['score'])
                            
                            # Course details
                            st.markdown(f"**Description:** {course['description']}")
                            
                            # Match reasons
                            if course['reasons']:
                                st.markdown("**🎯 Why this course matches you:**")
                                for reason in course['reasons']:
                                    st.markdown(f"• {reason}")
                            
                            # Modules
                            if course['modules'] and course['modules'] != 'No modules listed':
                                st.markdown(f"**📚 Key Modules:** {course['modules']}")
                            
                            # Requirements and fees
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("💰 Fees", course['fees'])
                            with col2:
                                st.metric("📚 Min GPA", course['min_gpa'])
                            with col3:
                                st.metric("🗣️ Min IELTS", course['min_ielts'])
                            
                            # Action buttons
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button(f"📋 More Info", key=f"more_info_{i}"):
                                    st.info(f"Course prospects: {course['career_prospects']}")
                            with col2:
                                if st.button(f"❤️ Save Course", key=f"save_course_{i}"):
                                    if course['course_name'] not in current_profile.preferred_courses:
                                        current_profile.preferred_courses.append(course['course_name'])
                                        st.session_state.ai_system.profile_manager.save_profile(current_profile)
                                        st.success("✅ Added to your favorites!")
                                    else:
                                        st.info("This course is already in your favorites.")
                            with col3:
                                if st.button(f"✉️ Apply Now", key=f"apply_now_{i}"):
                                    st.info(f"📧 Contact: {config.admissions_email}")
                        
                        st.markdown("---")
                
                else:
                    st.warning("❌ No course recommendations found. Please update your profile or try different preferences.")
                    
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {e}")


def render_admission_prediction():
    """Render admission prediction interface"""
    st.header("📊 Admission Probability Prediction")
    
    if not st.session_state.profile_active:
        st.warning("👤 Please create a profile to get admission predictions.")
        return
    
    current_profile = st.session_state.current_profile
    
    # Course selection for prediction
    courses_list = ["Computer Science BSc", "Business Management BA", "Data Science MSc", 
                   "Engineering BEng", "Psychology BSc"]
    selected_course = st.selectbox("🎯 Select Course for Prediction", courses_list)
    
    if st.button("🔮 Predict Admission Chances", key="predict_btn"):
        with st.spinner("🧠 Analyzing your profile and predicting admission probability..."):
            try:
                # Prepare profile data for prediction
                profile_data = current_profile.to_dict()
                profile_data['course_applied'] = selected_course
                
                prediction = st.session_state.ai_system.predictive_engine.predict_admission_probability(profile_data)
                
                # Display results
                probability = prediction['probability']
                confidence = prediction['confidence']
                
                # Probability display
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.metric("🎯 Admission Probability", f"{probability:.1%}")
                    
                    # Progress bar with color coding
                    if probability >= 0.7:
                        st.success(f"🎉 High chance of admission!")
                    elif probability >= 0.5:
                        st.warning(f"⚡ Moderate chance - room for improvement")
                    else:
                        st.error(f"📈 Lower chance - significant improvement needed")
                
                with col2:
                    st.metric("🎯 Confidence", confidence.title())
                with col3:
                    # Risk level
                    if probability >= 0.7:
                        risk = "Low Risk"
                    elif probability >= 0.5:
                        risk = "Medium Risk"
                    else:
                        risk = "High Risk"
                    st.metric("⚠️ Risk Level", risk)
                
                # Factors analysis
                st.subheader("📈 Key Factors Influencing Your Prediction")
                factors = prediction.get('factors', [])
                for factor in factors:
                    st.markdown(f"• {factor}")
                
                # Recommendations
                st.subheader("💡 Recommendations to Improve Your Chances")
                recommendations = prediction.get('recommendations', [])
                for rec in recommendations:
                    st.markdown(f"• {rec}")
                
                # Feature importance (if available)
                importance = prediction.get('feature_importance', {})
                if importance and PLOTLY_AVAILABLE:
                    st.subheader("📊 What Matters Most")
                    
                    # Create importance chart
                    importance_df = pd.DataFrame([
                        {"Factor": k.replace('_', ' ').title(), "Importance": v} 
                        for k, v in importance.items()
                    ]).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(importance_df, x='Importance', y='Factor', 
                               orientation='h', title="Admission Factors Importance")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error predicting admission: {e}")


def render_document_verification():
    """Render document verification interface"""
    st.header("📄 AI Document Verification")
    
    st.info("Upload your documents for AI-powered verification and analysis.")
    
    # Document type selection
    doc_type = st.selectbox("📋 Document Type", [
        "transcript", "ielts_certificate", "passport", 
        "personal_statement", "reference_letter"
    ])
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload Document", 
        type=['pdf', 'jpg', 'jpeg', 'png', 'doc', 'docx'],
        help="Supported formats: PDF, JPG, PNG, DOC, DOCX (Max 10MB)"
    )
    
    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Additional information form
        with st.form("document_verification"):
            st.subheader("📝 Document Information")
            
            if doc_type == "transcript":
                institution = st.text_input("Institution Name")
                graduation_date = st.date_input("Graduation Date")
                overall_grade = st.text_input("Overall Grade/GPA")
                additional_info = {"institution": institution, "graduation_date": str(graduation_date), "grade": overall_grade}
                
            elif doc_type == "ielts_certificate":
                test_date = st.date_input("Test Date")
                test_center = st.text_input("Test Center")
                overall_score = st.number_input("Overall Score", 0.0, 9.0, 6.5, 0.5)
                additional_info = {"test_date": str(test_date), "test_center": test_center, "overall_score": overall_score}
                
            elif doc_type == "passport":
                passport_number = st.text_input("Passport Number")
                nationality = st.text_input("Nationality")
                expiry_date = st.date_input("Expiry Date")
                additional_info = {"passport_number": passport_number, "nationality": nationality, "expiry_date": str(expiry_date)}
                
            else:
                additional_info = {"file_name": uploaded_file.name, "file_type": doc_type}
            
            if st.form_submit_button("🔍 Verify Document"):
                with st.spinner("🤖 AI is analyzing your document..."):
                    try:
                        # Simulate document processing
                        document_data = {
                            "file_name": uploaded_file.name,
                            "file_size": uploaded_file.size,
                            "file_type": uploaded_file.type,
                            **additional_info
                        }
                        
                        verification_result = st.session_state.ai_system.document_verifier.verify_document(
                            document_data, doc_type
                        )
                        
                        # Display results
                        status = verification_result['verification_status']
                        confidence = verification_result.get('confidence_score', 0.0)
                        
                        # Status display
                        col1, col2 = st.columns(2)
                        with col1:
                            if status == "verified":
                                st.success(f"✅ Document Verified")
                            elif status == "needs_review":
                                st.warning(f"⚠️ Needs Manual Review")
                            else:
                                st.error(f"❌ Verification Failed")
                        
                        with col2:
                            st.metric("🎯 Confidence Score", f"{confidence:.1%}")
                        
                        # Issues found
                        issues = verification_result.get('issues_found', [])
                        if issues:
                            st.subheader("⚠️ Issues Identified")
                            for issue in issues:
                                st.markdown(f"• {issue}")
                        
                        # Recommendations
                        recommendations = verification_result.get('recommendations', [])
                        if recommendations:
                            st.subheader("💡 Recommendations")
                            for rec in recommendations:
                                st.markdown(f"• {rec}")
                        
                        # Verified fields
                        verified_fields = verification_result.get('verified_fields', {})
                        if verified_fields:
                            st.subheader("📋 Field Verification")
                            
                            for field, data in verified_fields.items():
                                col1, col2, col3 = st.columns([2, 1, 1])
                                with col1:
                                    st.text(field.replace('_', ' ').title())
                                with col2:
                                    if data['verified']:
                                        st.success("✅ Verified")
                                    else:
                                        st.error("❌ Not Verified")
                                with col3:
                                    st.text(f"{data['confidence']:.1%}")
                        
                        # Store verification in profile
                        if st.session_state.profile_active:
                            current_profile = st.session_state.current_profile
                            current_profile.add_interaction("document_verification")
                            st.session_state.ai_system.profile_manager.save_profile(current_profile)
                        
                    except Exception as e:
                        st.error(f"❌ Verification error: {e}")


def render_analytics_dashboard():
    """Render analytics dashboard"""
    st.header("📈 Analytics Dashboard")
    
    # System overview
    status = st.session_state.ai_system.get_system_status()
    data_stats = status.get('data_stats', {})
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        courses_total = data_stats.get('courses', {}).get('total', 0)
        st.metric("🎓 Total Courses", courses_total)
    
    with col2:
        apps_total = data_stats.get('applications', {}).get('total', 0)
        st.metric("📝 Applications", apps_total)
    
    with col3:
        faqs_total = data_stats.get('faqs', {}).get('total', 0)
        st.metric("❓ FAQs Available", faqs_total)
    
    with col4:
        search_ready = data_stats.get('search_index', {}).get('search_ready', False)
        st.metric("🔍 Search Ready", "✅" if search_ready else "❌")
    
    # Data overview
    st.subheader("📊 Data Overview")
    
    # Course level distribution
    ai_system = st.session_state.ai_system
    if not ai_system.data_manager.courses_df.empty and PLOTLY_AVAILABLE:
        courses_df = ai_system.data_manager.courses_df
        
        if 'level' in courses_df.columns:
            level_counts = courses_df['level'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📚 Courses by Level")
                fig = px.pie(values=level_counts.values, names=level_counts.index, 
                           title="Course Distribution by Academic Level")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("💰 Fee Ranges")
                if 'fees_international' in courses_df.columns:
                    fig = px.histogram(courses_df, x='fees_international', 
                                     title="International Fee Distribution", nbins=10)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Application status distribution
    if not ai_system.data_manager.applications_df.empty and PLOTLY_AVAILABLE:
        apps_df = ai_system.data_manager.applications_df
        
        if 'status' in apps_df.columns:
            st.subheader("📈 Application Status Distribution")
            status_counts = apps_df['status'].value_counts()
            
            fig = px.bar(x=status_counts.index, y=status_counts.values,
                        title="Applications by Status", 
                        color=status_counts.values,
                        color_continuous_scale="viridis")
            st.plotly_chart(fig, use_container_width=True)
    
    # System performance
    st.subheader("⚡ System Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Component Status:**")
        for component, is_ready in status.items():
            if isinstance(is_ready, bool):
                icon = "✅" if is_ready else "❌"
                st.markdown(f"{icon} {component.replace('_', ' ').title()}")
    
    with col2:
        if st.session_state.profile_active:
            profile = st.session_state.current_profile
            st.markdown("**👤 Your Activity:**")
            st.markdown(f"• Interactions: {profile.interaction_count}")
            st.markdown(f"• Profile Completion: {profile.profile_completion:.0f}%")
            st.markdown(f"• Favorite Features: {', '.join(profile.favorite_features[:3])}")


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="UEL AI Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session
    if not init_streamlit_session():
        st.stop()
    
    # Render sidebar
    render_sidebar()
    
    # Handle different views based on session state
    if st.session_state.get('show_profile_creator', False):
        render_profile_creator()
        # "Back to Main" button from profile creator
        if st.button("← Back to Main", key="back_from_creator"):
            st.session_state.show_profile_creator = False
            st.session_state.show_login = False
            st.rerun()
    elif st.session_state.get('show_login', False) and not st.session_state.profile_active:
        render_login_form()
        # "Back to Main" button from login
        if st.button("← Back to Main", key="back_from_login"):
            st.session_state.show_login = False
            st.session_state.show_profile_creator = False
            st.rerun()
    else:
        render_main_interface()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        get_logger(__name__).error(f"Application startup error: {e}")
        st.info("Please refresh the page or contact support if the issue persists.")"""
UEL AI System - Streamlit Web Interface Module
"""

import time
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from config import config, PROFILE_DATA_DIR
from profile_manager import UserProfile
from main_system import UELAISystem
from utils import get_logger

# Try to import plotly
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def init_streamlit_session():
    """Initialize Streamlit session state"""
    if 'ai_system' not in st.session_state:
        try:
            with st.spinner("🚀 Initializing UEL AI System..."):
                st.session_state.ai_system = UELAISystem()
                st.session_state.system_ready = True
        except Exception as e:
            st.error(f"❌ Failed to initialize AI system: {e}")
            st.session_state.system_ready = False
            return False
    
    if 'current_profile' not in st.session_state:
        st.session_state.current_profile = None
        st.session_state.profile_active = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'feature_usage' not in st.session_state:
        st.session_state.feature_usage = defaultdict(int)
    
    return True


def render_sidebar():
    """Render application sidebar"""
    st.sidebar.title("🎓 UEL AI Assistant")
    st.sidebar.markdown("---")
    
    # System status
    if st.session_state.get('system_ready', False):
        status = st.session_state.ai_system.get_system_status()
        
        st.sidebar.subheader("🔧 System Status")
        
        # Status indicators
        status_indicators = {
            "🤖 AI Ready": status.get('system_ready', False),
            "🧠 LLM Available": status.get('ollama_available', False),
            "🎤 Voice Ready": status.get('voice_available', False),
            "📊 ML Models": status.get('ml_ready', False),
            "📚 Data Loaded": status.get('data_loaded', False)
        }
        
        for label, is_ready in status_indicators.items():
            icon = "✅" if is_ready else "❌"
            st.sidebar.markdown(f"{icon} **{label}**")
        
        st.sidebar.markdown("---")
    
    # Profile section
    st.sidebar.subheader("👤 Student Profile")
    
    if st.session_state.profile_active:
        profile = st.session_state.current_profile
        st.sidebar.success(f"Welcome, {profile.first_name}!")
        st.sidebar.metric("Profile Completion", f"{profile.profile_completion:.0f}%")
        
        if st.sidebar.button("📝 Edit Profile", key="sidebar_edit_profile"):
            st.info("Edit Profile functionality to be implemented.")
        
        if st.sidebar.button("🚪 Sign Out", key="sidebar_sign_out"):
            st.session_state.current_profile = None
            st.session_state.profile_active = False