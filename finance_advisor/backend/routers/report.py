# backend/routers/report.py

from fastapi import APIRouter, HTTPException, Query, Response
from finance_advisor.backend.memory.store import memory_store
import traceback

router = APIRouter(prefix="/download_plan", tags=["report"])


@router.get("")
def download_plan(session_id: str = Query(..., description="User session ID")):
    """
    Generates and returns a financial plan PDF using data stored in Redis memory.
    """

    try:
        # ---------------------------------
        # FETCH MEMORY FROM REDIS
        # ---------------------------------
        entity = memory_store.get_entity(session_id) or {}
        summary = memory_store.get_summary(session_id) or ""

        # Extract meaningful values
        age = entity.get("age")
        risk_category = entity.get("risk_category")
        last_portfolio = entity.get("last_portfolio")
        last_simulation = entity.get("last_simulation")

        # ---------------------------------
        # REPORTLAB SETUP
        # ---------------------------------
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch
        from io import BytesIO

        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)

        # ---------------------------------
        # HEADER
        # ---------------------------------
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(1 * inch, 10.5 * inch, "Financial Plan Summary")

        pdf.setFont("Helvetica", 11)
        pdf.drawString(1 * inch, 10.1 * inch, f"Session ID: {session_id}")

        y = 9.7 * inch

        # ---------------------------------
        # USER BASIC DETAILS
        # ---------------------------------
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(1 * inch, y, "User Information")
        y -= 0.3 * inch

        pdf.setFont("Helvetica", 11)

        if age:
            pdf.drawString(1 * inch, y, f"Age: {age}")
            y -= 0.25 * inch

        if risk_category:
            pdf.drawString(1 * inch, y, f"Risk Category: {risk_category}")
            y -= 0.25 * inch

        # ---------------------------------
        # PORTFOLIO SECTION
        # ---------------------------------
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(1 * inch, y, "Recommended Portfolio")
        y -= 0.3 * inch

        pdf.setFont("Helvetica", 11)

        if last_portfolio:
            for k, v in last_portfolio.items():
                pdf.drawString(1 * inch, y, f"{k}: {v}%")
                y -= 0.22 * inch
                if y < 1.5 * inch:
                    pdf.showPage()
                    y = 10.5 * inch
        else:
            pdf.drawString(1 * inch, y, "No portfolio data available.")
            y -= 0.3 * inch

        # ---------------------------------
        # SIMULATION RESULTS
        # ---------------------------------
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(1 * inch, y, "Simulation Results")
        y -= 0.3 * inch

        pdf.setFont("Helvetica", 11)

        if last_simulation:
            pdf.drawString(1 * inch, y, f"Expected Value: {last_simulation.get('expected_value')}")
            y -= 0.22 * inch
            pdf.drawString(1 * inch, y, f"Best Case: {last_simulation.get('best_case')}")
            y -= 0.22 * inch
            pdf.drawString(1 * inch, y, f"Worst Case: {last_simulation.get('worst_case')}")
            y -= 0.22 * inch
            pdf.drawString(1 * inch, y, f"Goal Achievement Probability: {last_simulation.get('probability_of_goal_achievement')}")
            y -= 0.3 * inch
        else:
            pdf.drawString(1 * inch, y, "No simulation data available.")
            y -= 0.3 * inch

        # ---------------------------------
        # CONVERSATION SUMMARY
        # ---------------------------------
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(1 * inch, y, "Conversation Summary")
        y -= 0.3 * inch

        pdf.setFont("Helvetica", 11)

        for line in summary.split("\n"):
            pdf.drawString(1 * inch, y, line[:100])  # avoid long overflow
            y -= 0.22 * inch
            if y < 1.5 * inch:
                pdf.showPage()
                y = 10.5 * inch

        pdf.showPage()
        pdf.save()

        pdf_bytes = buffer.getvalue()
        buffer.close()

        # ---------------------------------
        # RETURN PDF
        # ---------------------------------
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="financial_plan.pdf"'}
        )

    except Exception as ex:
        print("\n\n------------ REPORT ROUTER ERROR ------------")
        traceback.print_exc()
        print("----------------------------------------------\n\n")
        raise HTTPException(status_code=500, detail=str(ex))
