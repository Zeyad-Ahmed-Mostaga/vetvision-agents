# VetVision AI Agents Documentation

VetVision features a unified architecture with two distinct, role-specific AI agents built using LangGraph. Each agent is tailored to its specific audience—pet owners or veterinary doctors—with specialized workflows, safety guardrails, and toolsets.

---

## 1. User Agent (فيتو - Vito)

**Target Audience:** Pet Owners  
**Endpoint:** `POST /chat`  
**Model Architecture:** LangGraph ReAct Agent utilizing OpenRouter (Primary) with an automatic fallback to Gemini.

### 🧠 Workflow & Decision Engine

The User Agent operates as a warm, empathetic veterinary assistant, defaulting to an Egyptian Arabic personality but matching the user's language.

1. **Intent Recognition & Gateway:** Analyzes the user's input to determine the required action. If crucial information is missing, the agent triggers a strict gateway and pauses tool execution to ask for details:
   - *Medical Queries:* Requires the **animal type** (cat, dog, horse, other).
   - *Clinic/Hospital Search:* Requires the **user's location** (city/neighborhood).
   - *Medications:* Requires the exact **drug name**.
2. **Tool Routing:** Once all required information is gathered, the agent routes the query:
   - Uses RAG search for symptoms, diet, and general pet care.
   - Uses Web search for real-world locations, clinics, or specific commercial drug availability.
3. **Self-Correction:** If the RAG search returns no relevant information, the agent automatically falls back to the Web search tool to ensure the user receives a helpful answer.
4. **Response Generation:** Formulates a reassuring, practical response in the language the user wrote in.

### 🛠️ Capabilities & Tools

* **`vet_rag_search` (Advanced RAG):** Searches the proprietary VetVision knowledge base for trusted veterinary information.
* **`tavily_search` (Web Search):** An Egypt-aware web search tool used to find nearby veterinary clinics, up-to-date medication prices, and as a general fallback.

---

## 2. Vet Copilot Agent

**Target Audience:** Veterinary Doctors  
**Endpoint:** `POST /copilot/chat`  
**Model Architecture:** LangGraph Agent utilizing OpenRouter.

### 🧠 Workflow & Decision Engine

The Vet Copilot is a professional, concise, and highly strict medical assistant designed to help veterinarians manage patient records and retrieve clinical information.

1. **Identity & Context Initialization:** 
   - On the first interaction, the agent mandates asking for the doctor's name, which is remembered for the session.
   - The real current system date is dynamically injected into the agent's context to prevent date hallucination during visit logging.
2. **Strict Write-Safety Guardrails:** Differentiates between read-only medical lookups and database writes. The agent will **never** attempt to register or log a patient visit unless the doctor explicitly requests it.
3. **Patient Visit Flow:**
   - **First-Time Patient:** Gathers animal name, type, owner name, diagnosis, treatment, and optional weight. Registers the patient and issues a unique 6-character human-readable Animal ID.
   - **Returning Patient:** Asks for the existing Animal ID, verifies it, gathers the new visit's details, and logs the record.
4. **Professional Response:** Returns actionable, concise information and cites sources when providing medical advice.

### 🛠️ Capabilities & Tools

* **`vet_rag_search` (Medical Knowledge Lookup):** Queries the VetVision knowledge base for specific clinical symptoms, treatment protocols, and diseases.
* **`tavily_search` (Web Search):** Retrieves the latest veterinary research, protocols, and real-world drug data.
* **`register_first_visit` (Write DB):** Registers a new patient, logs their first visit, and generates a 6-character alphanumeric Animal ID.
* **`log_returning_visit` (Write DB):** Logs a new visit to an existing patient's medical record using their Animal ID.
* **`get_patient_history` (Read-Only DB):** Retrieves and summarizes the full medical history of a patient by their Animal ID.
* **`generate_patient_report` (Reporting):** Generates a professional, bilingual (Arabic/English) PDF medical report using a Chromium-rendered Playwright pipeline.
