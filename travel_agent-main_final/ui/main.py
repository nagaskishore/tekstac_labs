# ui/main.py - LEARNING PROJECT UI
# TODO: This shows placeholder responses to demonstrate UI structure
# Replace placeholder logic with actual AI agent integration when implementing

import streamlit as st
import requests
import json
import pandas as pd
from pathlib import Path
import sys

API_BASE_URL = "http://localhost:8000"

# Add project root to path for db_utils
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
try:
    from db import db_utils
except ImportError:
    from db import db_utils

# -----------------------------
# API helper functions (moved to top)
# -----------------------------
def plan_trip_api(user_input, user_id, trip_id, phase, approval_mode="manual", previous_state=None):
    try:
        phase = (phase or "").strip().lower()

        params = {
            "user_input": user_input,
            "user_id": user_id,
            "trip_id": trip_id if trip_id is not None else 0,
            "phase": phase,
        }

        # ? For phase4, also pass approval_mode and previous_state in JSON body
        if phase == "phase4_langgraph":
            params["approval_mode"] = approval_mode
            response = requests.post(
                f"{API_BASE_URL}/api/v1/plan_trip",
                params=params,
                json=previous_state,   # ? previous_state as body
                timeout=120
            )
        else:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/plan_trip",
                params=params,
                timeout=120
            )

        if response.status_code == 200:
            return response.json()

        st.error(f"API Error: {response.status_code}")
        st.code(response.text)
        return None

    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None


def approve_api(trip_id, user_id, approval, phase, thread_id=None, feedback=None):
    payload = {
        "trip_id": trip_id if trip_id is not None else 0,
        "user_id": user_id,
        "approval": approval,
        "phase": phase
    }
    if feedback:
        payload["feedback"] = feedback

    # ? Phase4 must send thread_id
    if phase == "phase4_langgraph":
        payload["thread_id"] = thread_id

    try:
        response = requests.post(f"{API_BASE_URL}/api/v1/approve", json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()

        st.error(f"API Error: {response.status_code}")
        st.code(response.text)
        return None

    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

st.set_page_config(page_title="TravelMate AI", layout="wide")

# Sidebar Navigation
st.sidebar.title("TravelMate AI")
page = st.sidebar.radio("Navigate", ["Welcome Page", "Trip Planner", "Database Viewer"])

# -----------------------------
# Welcome Page
# -----------------------------
if page == "Welcome Page":
    st.title("Welcome to TravelMate AI - Learning Project")
    st.markdown("""
    ## [LEARNING PROJECT] AI Multi-Agent Travel Planning System
    
    This is a **hands-on learning project** where you'll implement AI agents across different frameworks.
    
    ### What's Already Provided:
    - [PROVIDED] **Database layer** (`db/`) - Complete schema, models, and utilities
    - [PROVIDED] **API toolkits** (`toolkits/`) - Amadeus, weather, web search tools
    - [PROVIDED] **Configuration** (`config.py`) - API key management
    - [PARTIAL] **UI Interface** (this Streamlit app) - Template structure, you connect the agents
    - [PROVIDED] **Data models** (`api/datamodels.py`) - All Pydantic models
    
    ### Your Learning Tasks (Implement These):
    - **Phase 2:** Implement CrewAI agents in `phases/phase2_crewai/`
    - **Phase 3:** Implement AutoGen agents in `phases/phase3_autogen/`  
    - **Phase 4:** Implement LangGraph workflow in `phases/phase4_langgraph/`
    - **API Layer:** Complete FastAPI endpoints in `api/app.py`
    - **UI Connection:** Connect your agents to replace placeholder messages
    
    ### Learning Progression:
    1. **Phase 2 (CrewAI):** Sequential agent workflow - InfoCollector ? Planner ? Optimizer
    2. **Phase 3 (AutoGen):** Collaborative debate - Agents discuss and reach consensus  
    3. **Phase 4 (LangGraph):** Stateful workflows - Human-in-the-loop, persistence, recovery
    
    ### Test Users (Pre-loaded in Database):
    - **Alice:** Luxury traveler (art, fine dining)
    - **Bob:** Budget backpacker (adventure, local experiences) 
    - **Carla:** Family travel (safety, kid-friendly)
    - **David:** Business travel (efficiency, business amenities)
    - **Emma:** Student travel (budget, authentic experiences)
    
    **[NEXT STEP]** Go to Trip Planner ? Select User ? Choose "Start New Trip" ? Try chatting!
    """)

# -----------------------------
# Trip Planner (Chat-style, API-driven)
# -----------------------------
elif page == "Trip Planner":
    st.title("TravelMate Trip Planner")
    
    # API Status Check
    try:
        test_response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if test_response.status_code == 200:
            st.success("? API Server is running - Connect your agents to enable full functionality!")
        else:
            st.warning("?? API Server responded with error - Showing placeholder mode")
    except requests.RequestException:
        st.info("**Learning Mode:** API server is not running. UI shows placeholder responses to demonstrate functionality.")
    
    st.markdown("""
        Use the chat below to describe your trip in natural language.
        **Example Inputs:**

1. **Complete Query:**
    'I want to plan a leisure trip from Bangalore to Goa from December 15-18, 2025, for 2 adults with a budget of 8000 INR.'

2. **Multi-turn (2-turn) Query:**                  
    - Turn 1: 'I want to plan a solo business trip from Mumbai to Singapore.'
    - Turn 2: '"Next month for 4 days starting on the 15th with a budget of $1200 USD"'
""")
    from db import db_utils
    trip_id = None  # Initialize trip_id at the beginning
    with st.sidebar:
        st.header("Configuration")
        phase = st.selectbox("AI Phase", ["phase2_crewai", "phase3_autogen", "phase4_langgraph"])
        user_list = db_utils.get_all_users()
        if user_list:
            username = st.selectbox("User", user_list)
            user_id = db_utils.get_user_id_by_name(username)
            # Trip selection for user
            trips = db_utils.get_trips_by_user_name(username)
            trip_options = {f"{t['title']} (ID {t['id']})": t['id'] for t in trips} if trips else {}
            
            # Add "Start New Trip" option
            trip_options["Start New Trip"] = "new_trip"
            
            selected_trip_option = st.selectbox("Select Trip", list(trip_options.keys()))
            
            if selected_trip_option == "Start New Trip":
                trip_id = None
                st.info("[CHAT] Start chatting to create a new trip! Just describe your travel plans.")
            elif selected_trip_option:
                trip_id = trip_options[selected_trip_option]
                trip_details = db_utils.get_trip_with_plan(trip_id)
                if trip_details:
                    with st.expander("Trip Details", expanded=False):
                        st.write(f"**Trip:** {trip_details['title']}")
                        st.write(f"**Route:** {trip_details['origin']} to {trip_details['destination']}")
                        st.write(f"**Dates:** {trip_details['trip_startdate']} to {trip_details['trip_enddate']}")
                        st.write(f"**Status:** {trip_details['trip_status']}")
                        
                        # Show plan details if available
                        if trip_details.get('plan_status'):
                            st.write(f"**Plan Status:** {trip_details['plan_status']}")
                            st.write(f"**Plan Version:** {trip_details.get('plan_version', 'N/A')}")
                            if trip_details.get('total_estimated_cost'):
                                st.write(f"**Estimated Cost:** {trip_details['currency']} - {trip_details['total_estimated_cost']}")
                            
                            # Show detailed plan if available
                            try:
                                import json
                                if trip_details.get('hotels_json'):
                                    hotels = json.loads(trip_details['hotels_json'])
                                    st.write(f"**Hotels:** {len(hotels)} options")
                                if trip_details.get('flights_json'):
                                    flights = json.loads(trip_details['flights_json'])
                                    st.write(f"**Flights:** {len(flights)} options")
                            except:
                                pass
                        else:
                            st.write("**Plan Status:** No plan generated yet")
                # ? Phase4: reload persisted thread_id for approvals
            if phase == "phase4_langgraph" and trip_id:
                try:
                    persisted_thread_id = db_utils.get_latest_thread_id_for_trip(trip_id)
                    if persisted_thread_id:
                        st.session_state["thread_id"] = persisted_thread_id
                except Exception:
                    pass
        else:
            st.error("[LEARNING PROJECT] No users found in database. Run the database setup first!")
            username = None
            user_id = None
            trip_id = None

    # Chat history state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "phase" not in st.session_state:
        st.session_state["phase"] = []
    if "trip_id" not in st.session_state:
        st.session_state["trip_id"] = None
    if "plan" not in st.session_state:
        st.session_state["plan"] = None
    if "awaiting_approval" not in st.session_state:
        st.session_state["awaiting_approval"] = False
    if "context" not in st.session_state:
        st.session_state["context"] = {}
    if "last_approval_result" not in st.session_state:
        st.session_state["last_approval_result"]=None
    if "workflow_state" not in st.session_state:
        st.session_state["workflow_state"] = None  
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = None       
    if "next_action" not in st.session_state:
        st.session_state["next_action"] = None
    
    # Clear chat when user or trip selection changes or phase chnages

    if "current_selected_trip" not in st.session_state:
        st.session_state["current_selected_trip"] = None
    if "current_selected_user" not in st.session_state:
        st.session_state["current_selected_user"] = None
    if "current_selected_phase" not in st.session_state:
        st.session_state["current_selected_phase"] = None
    
    # Check if user or trip selection has changed
    current_trip_selection = trip_id if trip_id != "new_trip" else None
    current_user_selection = user_id if 'user_id' in locals() and user_id else None
    current_selected_phase= phase if  'phase' in locals() and phase else None
    
    if (st.session_state["current_selected_trip"] != current_trip_selection or 
        st.session_state["current_selected_user"] != current_user_selection or 
        st.session_state["current_selected_phase"] != current_selected_phase):
        # Clear chat history and related state when switching users or trips
        st.session_state["messages"] = []
        st.session_state["plan"] = None
        st.session_state["awaiting_approval"] = False
        st.session_state["context"]={}
        st.session_state["last_approval_result"]=None

        st.session_state["workflow_state"] = None
        st.session_state["thread_id"] = None
        st.session_state["next_action"] = None
        st.session_state["trip_id"] = None

        
        
        # Load historical chat for existing trips
        if current_trip_selection is not None and current_user_selection is not None:
            try:
                # Load chat history from database for this specific trip
                historical_messages = db_utils.load_chat_history(current_trip_selection)
                #st.write(historical_messages)
                if historical_messages:
                    # Convert database records to chat message format
                    for msg in historical_messages:
                        role = msg.get('role', 'assistant')  # role is already in correct format
                        st.session_state["messages"].append({
                            "role": role,
                            "content": msg.get('content', '')
                        })
                p_context=db_utils.get_trip_context(user_id=current_user_selection,trip_id=current_trip_selection,phase=current_selected_phase)
                
                if p_context["trip_status"]=='draft' and p_context["plan_status"] is None:
                    #option=st.selectbox(f"I can find a draft trip with no trip plans with mentioned trip id and phase\n,{trip_details} \nplease let me know shall i proceed with generating the trip plans for it",["--chose below options--","yes","no"])
                    
                    #if option=="yes":
                    st.session_state["context"]=p_context
                    #st.info(f"I am proceeding with flowing {trip_details} detais")
                    st.session_state["messages"].append({"role": "system", "content": json.dumps(p_context, indent=2)})
                    #with st.chat_message("user"):
                        #st.markdown(p_context)
                
               
            except Exception as e:
                # If historical chat loading fails, start with empty chat
                st.info(f"Starting fresh chat (historical messages unavailable: {str(e)})")
                st.session_state["messages"] = []
        
        # Update tracking variables
        st.session_state["current_selected_trip"] = current_trip_selection
        st.session_state["current_selected_user"] = current_user_selection
        st.session_state["current_selected_phase"]= current_selected_phase
        

    # Show chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["role"])
            st.markdown(msg["content"])

    #get trip context if already any draft trip for provided trip id
   

             
    prompt = st.chat_input("Ask your travel planner...")

    if prompt and user_id:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
                # Determine if this is for a new trip or existing trip
        # ? Use session trip_id if present (phase4 may create it after requirements complete)
        current_trip_id = st.session_state.get("trip_id")
        if current_trip_id is None:
            current_trip_id = trip_id  # fallback from sidebar selection
                
        prev_state = None
        approval_mode = "manual"
        plan=None

        if phase == "phase4_langgraph":
        # send previous_state if we already have workflow_state
            prev_state = st.session_state.get("workflow_state")

            plan = plan_trip_api(
                prompt,
                user_id,
                current_trip_id,
                phase,
                approval_mode=approval_mode,
                previous_state=prev_state
            )

            with st.expander("DEBUG: Phase4 State", expanded=False):
                    st.write("thread_id:", st.session_state.get("thread_id"))
                    st.write("has workflow_state:", st.session_state.get("workflow_state") is not None)
                    if st.session_state.get("workflow_state"):
                        st.json(st.session_state["workflow_state"])

        else:
            plan = plan_trip_api(
                prompt,
                user_id,
                current_trip_id,
                phase)
                
                # Show placeholder message when API is not available
        if plan is None:
                    # Different messages for new vs existing trips
            if current_trip_id is None:
                placeholder_message = f"""
                                            **[LEARNING PROJECT] NEW TRIP CREATION**

                                            Your request: *"{prompt}"*

                                            **To implement:** 
                                            - Choose your AI framework (Phase 2/3/4)
                                            - Implement agents in the respective phase folder
                                            - Connect agents to replace this placeholder

                                            **Hint:** Start with CrewAI Phase 2 for sequential agent workflow!
                                            """
            else:
                placeholder_message = f"""
                                            **[LEARNING PROJECT] TRIP MODIFICATION**

                                            Your request: *"{prompt}"*  
                                            Trip ID: {current_trip_id}

                                            **To implement:**
                                            - Load existing trip data using db_utils
                                            - Analyze modification request with your agents  
                                            - Update trip plan and save changes

                                            **Hint:** Use the same agents but with trip context!
                                            """
                    
                st.session_state["messages"].append({"role": "assistant", "content": placeholder_message})
                with st.chat_message("assistant"):
                    st.markdown(placeholder_message)
                    
                    # Don't set awaiting_approval for placeholder responses
                st.session_state["awaiting_approval"] = False
                st.session_state["plan"] = None
                
        else:
                    # TODO: Replace with actual API response processing from your agents
                    # Template structure showing expected response format
                    
                    
                    # Use the actual API response directly
            template_plan = plan
                    
            st.session_state["plan"] = template_plan
                    #st.markdown(template_plan)
                    
            if template_plan.get("success", False):
                st.session_state["trip_id"] = template_plan.get("trip_id", None)

                template_plan = plan
                st.session_state["plan"] = template_plan

                # ----------------------------
                # ? Phase4: ALWAYS persist state if returned (even if success=False)
                # ----------------------------
                if phase == "phase4_langgraph":
                    if template_plan.get("workflow_state") is not None:
                        st.session_state["workflow_state"] = template_plan.get("workflow_state")
                    if template_plan.get("thread_id") is not None:
                        st.session_state["thread_id"] = template_plan.get("thread_id")
                    if template_plan.get("next_action") is not None:
                        st.session_state["next_action"] = template_plan.get("next_action")

                    # awaiting approval only when orchestrator says so
                    st.session_state["awaiting_approval"] = (template_plan.get("next_action") == "awaiting_approval")
                else:
                    # Phase2/3: keep your logic
                    st.session_state["awaiting_approval"] = bool(template_plan.get("success", False))

# ? Persist trip_id if returned (phase4 trip_id may be 0 until complete)
                if template_plan.get("trip_id") is not None:
                    st.session_state["trip_id"] = template_plan.get("trip_id")
            else:
                st.session_state["awaiting_approval"] = False

            if phase=='phase2_crewai':
                # Show assistant response (simple summary)
                summary_lines = []
                if template_plan.get("success"):
                    req = template_plan.get("requirements", {})
                    summary_lines.append(f"**Trip:** {req.get('origin', '?')} to {req.get('destination', '?')} ({req.get('trip_startdate', '?')} to {req.get('trip_enddate', '?')})")
                    summary_lines.append(f"**Travelers:** {req.get('no_of_adults', 1)} adults, {req.get('no_of_children', 0)} children")
                    summary_lines.append(f"**Budget:** {req.get('budget', '?')} {req.get('currency', 'USD')}")
                    plan_data = template_plan.get("plan", {})
                    if plan_data.get("itinerary"):
                        summary_lines.append(f"**Itinerary:** {plan_data['itinerary']}")
                    if plan_data.get("hotels") and isinstance(plan_data["hotels"], list) and plan_data["hotels"]:
                        hotel = plan_data["hotels"][0]
                        summary_lines.append(f"**Hotel:** {hotel.get('name', 'N/A')} ({hotel.get('location', 'N/A')})")
                    if plan_data.get("flights") and isinstance(plan_data["flights"], list) and plan_data["flights"]:
                        flight = plan_data["flights"][0]
                        summary_lines.append(f"**Flight:** {flight.get('airline', 'N/A')} ({flight.get('departure_time', 'TBD')} to {flight.get('arrival_time', 'TBD')})")
                        opt = template_plan.get("optimization", {})
                    if opt.get("recommendations"):
                        summary_lines.append("**Recommendations:**")
                        for rec in opt["recommendations"]:
                            summary_lines.append(f"- {rec}")
                    else:
                        summary_lines.append(str(template_plan))
                assistant_content = "\n".join(summary_lines)
                st.session_state["messages"].append({"role": "assistant", "content": assistant_content})
                with st.chat_message("assistant"):
                    st.markdown(assistant_content)  
            if phase=='phase3_autogen':
                summary_lines = []
                if template_plan.get("success"):
                    req = template_plan.get("requirements", {})
                    summary_lines.append(f"**Trip:** {req.get('origin', '?')} to {req.get('destination', '?')} ({req.get('trip_startdate', '?')} to {req.get('trip_enddate', '?')})")
                    summary_lines.append(f"**Travelers:** {req.get('no_of_adults', 1)} adults, {req.get('no_of_children', 0)} children")
                    summary_lines.append(f"**Budget:** {req.get('budget', '?')} {req.get('currency', 'USD')}")
                    plan_data = template_plan.get("consensus_plan", {})
                    if plan_data.get("itinerary"):
                        summary_lines.append(f"**Itinerary:** {plan_data['itinerary']}")
                    if plan_data.get("hotels") and isinstance(plan_data["hotels"], list) and plan_data["hotels"]:
                        hotel = plan_data["hotels"][0]
                        summary_lines.append(f"**Hotel:** {hotel.get('name', 'N/A')} ({hotel.get('location', 'N/A')})")
                    if plan_data.get("flights") and isinstance(plan_data["flights"], list) and plan_data["flights"]:
                        flight = plan_data["flights"][0]
                        summary_lines.append(f"**Flight:** {flight.get('airline', 'N/A')} ({flight.get('departure_time', 'TBD')} to {flight.get('arrival_time', 'TBD')})")
                        opt = template_plan.get("optimizer_result", {})
                    if opt.get("recommendations"):
                        summary_lines.append("**Recommendations:**")
                        for rec in opt["recommendations"]:
                            summary_lines.append(f"- {rec}")
                    else:
                        summary_lines.append(str(template_plan))
                assistant_content = "\n".join(summary_lines)
                st.session_state["messages"].append({"role": "assistant", "content": assistant_content})
                with st.chat_message("assistant"):
                    st.markdown(assistant_content)  
            if phase == "phase4_langgraph":
                assistant_msg = template_plan.get("message", "No message")
                st.session_state["messages"].append({"role": "assistant", "content": assistant_msg})
                with st.chat_message("assistant"):
                    st.markdown(assistant_msg)

                # Optionally show travel plan summary if present
                if template_plan.get("travel_plan"):
                    with st.expander("Travel Plan", expanded=False):
                        st.json(template_plan["travel_plan"])

                if template_plan.get("optimization_result"):
                    with st.expander("Optimization Result", expanded=False):
                        st.json(template_plan["optimization_result"])

            
       

           


    # Approval logic (only show when API is working and plan is available)
    if st.session_state["awaiting_approval"] and st.session_state["plan"] and user_id:
        st.write("#### Do you approve this plan?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve"):
                st.session_state["messages"].append({"role":"user","content":"Approve the processed plan"})
                with st.chat_message("user"):
                    st.markdown("Approve the processed plan")   

                approve_result = approve_api(trip_id=st.session_state["trip_id"],
                                        user_id=user_id,
                                        approval=True,
                                        phase=phase,
                                        thread_id=st.session_state.get("thread_id"),
                                        feedback=None
                                    )

                st.session_state["last_approval_result"] = approve_result
                with st.chat_message("assistant"):
                    st.markdown("Approve result:\n" + json.dumps(approve_result, indent=2))

                if approve_result and approve_result.get("success") is True:
                    st.success("Trip approved!")
                    st.session_state["awaiting_approval"] = False
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": "Approve result:\n" + json.dumps(approve_result, indent=2)
                    })
                else:
                    st.error((approve_result or {}).get("message", "Approve failed."))

                st.rerun()
        with col2:
            feedback = st.text_input("Feedback for revision", key="feedback")
            if st.button("Reject/Revise"):
                st.session_state["messages"].append({"role":"user","content":"Rejecting this plan for the reason: "+feedback})
                with st.chat_message("user"):
                    st.markdown("Rejecting this plan for the reason"+feedback)  
                reject_result = approve_api(
                                            trip_id=st.session_state["trip_id"],
                                            user_id=user_id,
                                            approval=False,
                                            phase=phase,
                                            thread_id=st.session_state.get("thread_id"),
                                            feedback=feedback
                                        )
                st.session_state["last_approval_result"] = reject_result
                with st.chat_message("assistant"):
                    st.markdown("Approve result:\n" + json.dumps(reject_result, indent=2))

                if reject_result and reject_result.get("success") is True:
                    st.warning("Plan rejected. Status updated to 'rejected'.")
                    st.session_state["awaiting_approval"] = False

                    
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": "Reject/Revise result:\n" + json.dumps(reject_result, indent=2)
                    })
                else:
                    st.error((reject_result or {}).get("message", "Reject/Revise failed."))

                st.rerun()
    
    


    # Raw API output
    if st.session_state["plan"]:
        with st.expander("Raw API Output", expanded=False):
            st.json(st.session_state["plan"])
    # Raw API response for user feedback
    if st.session_state.get("last_approval_result") is not None:
        with st.expander("Last Approval/Reject Action Output",expanded=False):
            st.json(st.session_state["last_approval_result"])

# -----------------------------
# Database Viewer
# -----------------------------
elif page == "Database Viewer":
    st.title("Database Tables")

    tables = ["users", "trips", "trip_plans", "chat_history"]
    selected_table = st.selectbox("Select a table", tables)

    try:
        df = db_utils.load_table_as_dataframe(selected_table)
        st.write(df)
        if not df.empty:
            st.dataframe(df.astype(str), width='stretch')
            st.subheader("Table Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                if selected_table == "trips":
                    active_trips = len(df[df['trip_status'].isin(['draft', 'confirmed', 'in_progress'])])
                    st.metric("Active Trips", active_trips)
                elif selected_table == "trip_plans":
                    if 'status' in df.columns:
                        approved_plans = len(df[df['status'] == 'approved'])
                        st.metric("Approved Plans", approved_plans)
                    else:
                        st.metric("Total Plans", len(df))
                elif selected_table == "chat_history":
                    recent_messages = len(df[df['created_at'] > pd.Timestamp.now() - pd.Timedelta(days=7)])
                    st.metric("Recent Messages (7d)", recent_messages)
                else:
                    st.metric("Recent Entries", len(df))
        else:
            st.warning(f"Table {selected_table} is empty or could not be loaded. Check database setup.")
    except Exception as e:
        st.error(f"[LEARNING PROJECT] Could not load table {selected_table}: {e}")
    if st.button("Refresh Data"):
        st.rerun()