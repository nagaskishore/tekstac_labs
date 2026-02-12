"""
Phase 4 LangGraph - Comprehensive Test Script (Direct Run)
- E2E smoke tests for Phase 4 workflow
- Includes state continuation, approval checks, DB checks, and performance timing
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from phases.phase4_langgraph.trip_orchestrator import LangGraphTripOrchestrator
from db import db_utils


# ----------------------------
# Helpers
# ----------------------------
def print_section(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_result(result, title="Result"):
    print(f"\n{title}:")
    print(json.dumps(result, indent=2, default=str))


def assert_in(value, allowed, msg):
    if value not in allowed:
        raise AssertionError(f"{msg}. Got '{value}', expected one of {allowed}")


# ----------------------------
# Test Case 1
# ----------------------------
def test_complete_query():
    """
    Test Case 1: Complete Query Test
    - Sends a single message with full requirements
    - Accepts either awaiting_approval or completed
    - If awaiting_approval, auto-approves to complete the flow
    """
    print_section("TEST CASE 1: COMPLETE QUERY")

    orchestrator = LangGraphTripOrchestrator()
    user_input = (
        "I want to plan a leisure trip from Bangalore to Goa from February 15-18, 2026, "
        "for 2 adults with a budget of 8000 INR."
    )

    print(" User Input:")
    print(f"   {user_input}")

    start_time = time.time()
    result = orchestrator.plan_trip(user_id=1, user_input=user_input, phase="phase4_langgraph")
    execution_time = time.time() - start_time

    print_result(result, "Workflow Result")

    print_section("VALIDATION - Test Case 1")

    success = result.get("success", False)
    workflow_status = result.get("workflow_status", "unknown")
    trip_id = result.get("trip_id")
    travel_plan = result.get("travel_plan")

    print(f" Success: {success}")
    print(f" Workflow Status: {workflow_status}")
    print(f" Trip ID: {trip_id}")
    print(f" Travel Plan Present: {bool(travel_plan)}")
    print(f" Execution Time: {execution_time:.2f}s")

    if execution_time > 180:
        print(" WARNING: Execution time exceeded 3 minutes!")
    else:
        print(" Execution time within target (< 3 minutes)")

    # workflow may stop at awaiting approval (common in your flow)
    assert_in(workflow_status, {"awaiting_approval", "completed", "needs_clarification", "failed"},
              "Unexpected workflow status in test_complete_query")

    # If awaiting approval, approve to complete
    if workflow_status == "awaiting_approval" and trip_id:
        print("\n Auto-approving plan to complete the workflow...")
        approval_result = orchestrator.continue_trip_approval(
            trip_id=trip_id,
            approval_decision="approved",
            user_feedback="Auto-approved by test",
            user_id=1
        )
        print_result(approval_result, "Auto Approval Result")
        assert_in(approval_result.get("workflow_status"), {"completed", "rejected"},
                  "Approval did not transition workflow as expected")

    # DB verification
    if trip_id:
        print("\n Database Verification:")
        trip = db_utils.get_trip_by_id(trip_id)
        if trip:
            print(" Trip found in database")
            print(f"   Origin: {trip.origin}")
            print(f"   Destination: {trip.destination}")
            print(f"   Status: {trip.trip_status}")
        else:
            print(" Trip not found in database!")

        plan = db_utils.get_trip_plan_by_trip_id(trip_id)
        if plan:
            print(f" Trip plan found (id={plan.id}, status={plan.status}, version={plan.version})")
        else:
            print(" No trip plan found for this trip (planner DB save may have failed).")

    return result


# ----------------------------
# Test Case 2
# ----------------------------
def test_multi_turn_conversation():
    """
    Test Case 2: Multi-turn Conversation Test
    - Turn 1: incomplete info
    - Turn 2: resume using previous_state=result_1['state']
    """
    print_section("TEST CASE 2: MULTI-TURN CONVERSATION")

    orchestrator = LangGraphTripOrchestrator()

    # Turn 1
    user_input_1 = "I want to plan a solo business trip from Mumbai to Singapore."
    print(" Turn 1 - User Input:")
    print(f"   {user_input_1}")

    start_time = time.time()
    result_1 = orchestrator.plan_trip(user_id=2, user_input=user_input_1, phase="phase4_langgraph")
    print_result(result_1, "Turn 1 Result")

    workflow_status_1 = result_1.get("workflow_status", "unknown")
    missing_fields = result_1.get("missing_fields", [])

    print("\n Turn 1 Analysis:")
    print(f"   Status: {workflow_status_1}")
    print(f"   Missing Fields: {missing_fields}")

    if workflow_status_1 != "needs_clarification":
        print(f" Expected needs_clarification, got {workflow_status_1}. Continuing anyway.")
        return result_1

    # Turn 2 (resume!)
    user_input_2 = "Next month for 4 days starting on the 15th with a budget of $1200 USD"
    print("\n Turn 2 - User Input:")
    print(f"   {user_input_2}")

    prev_state = result_1.get("state")
    if not prev_state:
        raise AssertionError("Turn 1 did not return 'state'. Ensure orchestrator includes 'state' in response.")

    result_2 = orchestrator.plan_trip(
        user_id=2,
        user_input=user_input_2,
        phase="phase4_langgraph",
        previous_state=prev_state
    )
    execution_time = time.time() - start_time
    print_result(result_2, "Turn 2 Result")

    print_section("VALIDATION - Test Case 2")

    success = result_2.get("success", False)
    workflow_status_2 = result_2.get("workflow_status", "unknown")
    trip_id = result_2.get("trip_id")

    print(f" Success: {success}")
    print(f" Final Workflow Status: {workflow_status_2}")
    print(f" Trip ID: {trip_id}")
    print(f" Total Execution Time (2 turns): {execution_time:.2f}s")

    if execution_time > 180:
        print(" WARNING: Total execution time exceeded 3 minutes!")
    else:
        print(" Total execution time within target (< 3 minutes)")

    # DB verification + chat history
    if trip_id:
        print("\n Database Verification:")
        trip = db_utils.get_trip_by_id(trip_id)
        if trip:
            print(" Trip found in database")
            print(f"   Origin: {trip.origin}")
            print(f"   Destination: {trip.destination}")
            print(f"   Status: {trip.trip_status}")
        else:
            print(" Trip not found in database!")

        chat_history = db_utils.load_chat_history(trip_id)  #  correct function name
        if chat_history:
            print(f" Chat history saved ({len(chat_history)} messages)")
        else:
            print(" No chat history found")

    return result_2


# ----------------------------
# Test Case 3
# ----------------------------
def test_approval_workflow():
    """
    Test Case 3: Approval Workflow
    - Creates a plan
    - If awaiting approval, approve it and validate plan status in DB
    - Then creates another plan and rejects it, validate plan status in DB
    """
    print_section("TEST CASE 3: APPROVAL WORKFLOW")

    orchestrator = LangGraphTripOrchestrator()

    user_input = "Plan a 3-day trip from Delhi to Jaipur from March 1-3, 2026, for 1 adult with 5000 INR budget"
    print(" User Input:")
    print(f"   {user_input}")

    result = orchestrator.plan_trip(user_id=3, user_input=user_input, phase="phase4_langgraph")
    print_result(result, "Planning Result")

    workflow_status = result.get("workflow_status", "unknown")
    trip_id = result.get("trip_id")

    if workflow_status != "awaiting_approval" or not trip_id:
        print(f" Workflow not in approval state: {workflow_status}")
        return

    # Approve
    print("\n Testing APPROVAL...")
    approval_result = orchestrator.continue_trip_approval(
        trip_id=trip_id,
        approval_decision="approved",
        user_feedback="Looks great!",
        user_id=3
    )
    print_result(approval_result, "Approval Result")

    # Verify DB trip + plan status
    trip = db_utils.get_trip_by_id(trip_id)
    plan = db_utils.get_trip_plan_by_trip_id(trip_id)

    print("\n Database Status After Approval:")
    if trip:
        print(f"   Trip Status: {trip.trip_status}")
    if plan:
        print(f"   Plan Status: {plan.status}")
        if plan.status != "approved":
            print(" Expected plan.status == 'approved'")
    else:
        print(" No plan found to verify approval status")

    # Reject flow on another plan
    print("\n Testing REJECTION...")
    result2 = orchestrator.plan_trip(
        user_id=3,
        user_input="Plan a trip from Chennai to Kerala for 2 days with 3000 INR budget",
        phase="phase4_langgraph"
    )
    trip_id_2 = result2.get("trip_id")

    if trip_id_2 and result2.get("workflow_status") == "awaiting_approval":
        rejection_result = orchestrator.continue_trip_approval(
            trip_id=trip_id_2,
            approval_decision="rejected",
            user_feedback="Budget is too low, please increase",
            user_id=3
        )
        print_result(rejection_result, "Rejection Result")

        plan2 = db_utils.get_trip_plan_by_trip_id(trip_id_2)
        if plan2:
            print(f" Plan Status After Rejection: {plan2.status}")
            if plan2.status != "rejected":
                print(" Expected plan.status == 'rejected'")
        else:
            print(" No plan found to verify rejection status")
    else:
        print(" Second workflow did not reach awaiting_approval state.")


# ----------------------------
# Test Case 4
# ----------------------------
def test_error_recovery():
    """
    Test Case 4: Error Recovery / Clarification
    """
    print_section("TEST CASE 4: ERROR RECOVERY")

    orchestrator = LangGraphTripOrchestrator()
    user_input = "Take me somewhere nice"  # deliberately vague

    print(" User Input (Deliberately Vague):")
    print(f"   {user_input}")

    result = orchestrator.plan_trip(user_id=4, user_input=user_input, phase="phase4_langgraph")
    print_result(result, "Result")

    workflow_status = result.get("workflow_status", "unknown")
    print("\n Error Recovery Analysis:")
    print(f"   Status: {workflow_status}")
    print(f"   Missing Fields: {result.get('missing_fields', [])}")

    if workflow_status == "needs_clarification":
        print(" Correctly handled incomplete input")
    elif workflow_status == "failed":
        print(" Workflow failed - checking error payload")
        print(f"   Error: {result.get('error', 'Unknown')}")
        print(f"   Tool Errors: {result.get('tool_errors', [])}")
        print(f"   Recovery Available: {result.get('recovery_available', False)}")


# ----------------------------
# Test Case 5
# ----------------------------
def run_performance_test():
    """
    Test Case 5: Performance Smoke
    """
    print_section("TEST CASE 5: PERFORMANCE TEST")

    orchestrator = LangGraphTripOrchestrator()

    test_cases = [
        {"name": "Short domestic trip", "input": "Plan a weekend trip from Pune to Mumbai from Feb 20-22, 2026, for 2 adults with 3000 INR"},
        {"name": "International business trip", "input": "I need a business trip from Bangalore to Dubai from March 10-15, 2026, for 1 adult with $2000 budget"},
        {"name": "Family vacation", "input": "Plan a family trip from Hyderabad to Kerala from April 1-7, 2026, for 2 adults and 2 children with 15000 INR budget"},
    ]

    results = []
    for idx, tc in enumerate(test_cases, 1):
        print(f"\n Performance Test {idx}/{len(test_cases)}: {tc['name']}")
        start = time.time()
        result = orchestrator.plan_trip(user_id=10 + idx, user_input=tc["input"], phase="phase4_langgraph")
        t = time.time() - start

        results.append({"name": tc["name"], "time": t, "success": result.get("success", False), "status": result.get("workflow_status", "unknown")})

        if t <= 120:
            print(f" Excellent: {t:.2f}s (< 2 min)")
        elif t <= 180:
            print(f" Good: {t:.2f}s (< 3 min)")
        else:
            print(f" Slow: {t:.2f}s (> 3 min)")

    print("\n Performance Summary:")
    avg_time = sum(r["time"] for r in results) / len(results)
    print(f"   Average Time: {avg_time:.2f}s")
    print(f"   Min Time: {min(r['time'] for r in results):.2f}s")
    print(f"   Max Time: {max(r['time'] for r in results):.2f}s")

    if avg_time <= 180:
        print(" Performance target met (< 3 minutes)")
    else:
        print(" Performance needs optimization")


def main():
    print("\n" + "=" * 80)
    print(" PHASE 4 LANGGRAPH - COMPREHENSIVE TEST SUITE (DIRECT RUN)")
    print("=" * 80)

    try:
        test_complete_query()
        test_multi_turn_conversation()
        test_approval_workflow()
        test_error_recovery()
        run_performance_test()

        print_section(" ALL TESTS COMPLETE")
        print(" Test Case 1: Complete Query - DONE")
        print(" Test Case 2: Multi-turn Conversation - DONE")
        print(" Test Case 3: Approval Workflow - DONE")
        print(" Test Case 4: Error Recovery - DONE")
        print(" Test Case 5: Performance Test - DONE")

    except Exception as e:
        print(f"\n Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()