import streamlit as st
import fitz  # PyMuPDF
import re
import json

# -------------------- MODEL LOADER --------------------
class ModelLoader:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            from groq import Groq
            cls._client = Groq(api_key=st.secrets["API_KEY"])
        return cls._client

# -------------------- EXTRACTOR --------------------
class Extractor:
    def __init__(self, text):
        self.text = text
        self.client = ModelLoader.get_client()

    def extract_skills(self):
        prompt = f"""
Extract only technical skills, programming languages, frameworks, tools, certifications, and key technical terms
from the following text. Return as a comma-separated list, lowercase, no explanations.
Ignore non-technical terms, job location, benefits, or experience descriptions.

Text:
{self.text}
"""
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # smaller, faster model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_completion_tokens=1024,
                top_p=0.95,
                stream=False
            )
            raw_skills = completion.choices[0].message.content
        except Exception as e:
            st.error(f"LLM extraction failed: {e}")
            raw_skills = ""

        return self.clean_skills(raw_skills)

    @staticmethod
    def clean_skills(raw_text):
        skills = []
        for word in raw_text.split(','):
            word = word.strip().lower()
            word = re.sub(r'[^a-zA-Z0-9/\s]', '', word)
            if '/' in word:
                skills.extend([w.strip() for w in word.split('/') if len(w.strip()) > 2])
            elif len(word) > 2:
                skills.append(word)
        return list(dict.fromkeys(skills))  # remove duplicates

# -------------------- SYNONYM MAPPER --------------------
class SynonymMapper:
    def __init__(self):
        self.client = ModelLoader.get_client()
        self.cache = {}

    def get_synonyms_batch(self, skills):
        uncached = [s for s in skills if s not in self.cache]
        if uncached:
            prompt = f"""
Provide synonyms for these technical skills as a JSON dictionary.
Each skill should be a key, and the value a list of common abbreviations, alternate names, and synonyms.
Return valid JSON only.

Skills: {uncached}
"""
            try:
                completion = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_completion_tokens=2048,
                    top_p=0.95,
                    stream=False
                )
                raw = completion.choices[0].message.content
                data = json.loads(raw)
            except Exception:
                data = {s: [s] for s in uncached}  # fallback
            self.cache.update(data)
        return {s: self.cache[s] for s in skills}

# -------------------- ADVISOR --------------------
class Advisor:
    def __init__(self, resume_skills, jd_skills, syn_mapper):
        self.resume_skills = resume_skills
        self.jd_skills = jd_skills
        self.syn_mapper = syn_mapper

    def compare(self):
        all_skills = list(set(self.resume_skills + self.jd_skills))
        syn_dict = self.syn_mapper.get_synonyms_batch(all_skills)

        resume_set = set()
        for skill in self.resume_skills:
            resume_set.update(syn_dict[skill])

        matching = []
        missing = []

        for jd_skill in self.jd_skills:
            jd_syns = set(syn_dict[jd_skill])
            if resume_set & jd_syns:
                matching.append(jd_skill)
            else:
                missing.append(jd_skill)

        ats_score = len(matching) / len(self.jd_skills) * 100 if self.jd_skills else 0
        return matching, missing, ats_score

# -------------------- PDF TO TEXT --------------------
def pdf_to_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

# -------------------- STREAMLIT APP --------------------
def main():
    st.set_page_config(page_title="Hackathon Resume-JD ATS", layout="wide")
    st.title("📄 Hackathon Resume-JD ATS Analyzer")

    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    jd_text = st.text_area("Paste Job Description", height=300)

    if st.button("Analyze") and resume_file and jd_text:
        with st.spinner("Extracting skills and comparing..."):
            resume_text = pdf_to_text(resume_file)

            resume_skills = Extractor(resume_text).extract_skills()
            jd_skills = Extractor(jd_text).extract_skills()

            syn_mapper = SynonymMapper()
            advisor = Advisor(resume_skills, jd_skills, syn_mapper)
            matching, missing, ats_score = advisor.compare()

        # Results
        st.subheader("📊 Analysis Results")
        st.metric("ATS Score", f"{ats_score:.1f}%")
        st.metric("Matching Skills", len(matching))
        st.metric("Missing Skills", len(missing))
        st.metric("JD Skills Found", f"{len(matching)/len(jd_skills)*100:.1f}%" if jd_skills else 0)

        st.subheader("🔹 Matching Skills")
        st.write(matching)
        st.subheader("🔹 Missing Skills")
        st.write(missing)

if __name__ == "__main__":
    main()
