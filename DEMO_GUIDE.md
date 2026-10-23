# VoiceOps 2026: 2-Minute Demo Video Walkthrough Guide

This guide provides the exact script and instructions to record the **2-minute live demo** using our Streamlit web application (`streamlit run app.py`) or the voice pipeline.

---

## Demo Video Requirements (from Capstone Brief)
- **Maximum Duration:** 2 minutes (120 seconds).
- **Core Elements Required:**
  1. A question answered directly from the company documents.
  2. A conversational follow-up (remembering context).
  3. One human handover (escalation) OR one out-of-scope question ("I don't know").
  4. Spoken customer voice input and Gnani TTS voice response.

---

## 2-Minute Demo Script & Timeline

### Part 1: Launch Application & Opening (0:00 - 0:20)
- **Visual:** Open browser to `http://localhost:8501`. Show the clean ZentroMart AI Customer Support interface with Gnani voice options and real-time telemetry.
- **Audio/Spoken:**  
  *Presenter:* "Welcome to the VoiceOps 2026 demo for ZentroMart, built for the IIT Guwahati × Gnani.ai Capstone. Here is our grounded conversational support agent powered by Gnani speech models."
- **Customer Turn 1:** Click Microphone or type:  
  *Customer:* *"Hello, mujhe apne order ke baare mein help chahiye."*
- **Agent Response (Spoken via Gnani TTS):**  
  *Agent:* *"नमस्ते! मैं ZentroMart का AI सहायता सहायक हूँ। कृपया अपना 8 अंकों का ऑर्डर नंबर बताएँ।"*
- **Highlight:** Agent introduces itself as an AI assistant as mandated by STY-1.

---

### Part 2: Question Grounded in Documents + Verification (0:20 - 0:50)
- **Customer Turn 2:**  
  *Customer:* *"Order number hai 8978 2837."*
- **Agent Response (IDV Gated):**  
  *Agent:* *"For your security, please verify by providing the last 4 digits of your registered phone or delivery pincode."*
- **Highlight:** Show real-time telemetry in right column: `Verification Status: PENDING_PROOF`.
- **Customer Turn 3 (Proof Provided):**  
  *Customer:* *"Last four digits 4533 hain."*
- **Agent Response:**  
  *Agent:* *"Thank you, verification successful! I see your order 89782837. How can I assist you with it today?"*

---

### Part 3: Contextual Follow-Up & Policy Reasoning (0:50 - 1:20)
- **Customer Turn 4 (Policy Query):**  
  *Customer:* *"Mera order already ship ho gaya hai. Kya main ise cancel kar sakta hoon?"*
- **Agent Response (CAN-2 & REF-1):**  
  *Agent:* *"A shipped order cannot be cancelled directly. You can refuse delivery at your door, and your refund will be processed within 7 to 10 days once returned."*
- **Customer Turn 5 (Follow-up without repeating context):**  
  *Customer:* *"And how long will the refund take to my UPI?"*
- **Agent Response (REF-2):**  
  *Agent:* *"Refunds to UPI take 1 to 3 business days from the initiation date."*
- **Highlight:** The agent remembers the order and UPI payment method seamlessly.

---

### Part 4: High-Value Escalation Handover & Ticket Creation (1:20 - 1:50)
- **Customer Turn 6 (Damage on high-value item):**  
  *Customer:* *"Maine Zentro Vista 43-inch 4K Smart TV damage report kiya tha. Abhi replacement approve kar do na?"*
- **Agent Response (DWM-3, ESC-1, ESC-4):**  
  *Agent:* *"Under our policy, claims for high-value items must be reviewed by our Priority Resolution team. I have created a ticket, and a colleague will reach out to you."*
- **Highlight:** Switch to the **Escalation Tickets** tab in Streamlit and show the newly created ticket (`TKT...`) logged with summary and escalation reason `high_value_damage`.

---

### Part 5: Out-of-Scope Boundary Check ("I don't know") (1:50 - 2:00)
- **Customer Turn 7:**  
  *Customer:* *"Can I schedule a delivery at midnight?"*
- **Agent Response:**  
  *Agent:* *"Sorry, I don't have information about that in our policies. I can connect you to a human colleague who can help."*
- **Closing:**  
  *Presenter:* *"The agent never hallucinates policies, strictly adheres to company documents, and provides an end-to-end voice experience with Gnani.ai. Thank you!"*

---

## Instructions to Record
1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Use OBS Studio, Windows Xbox Game Bar (`Win + G`), or screen recorder.
3. Record the screen and computer audio so Gnani TTS spoken output is clearly audible.
4. Export as `demo.mp4` and place in project root for packaging.
