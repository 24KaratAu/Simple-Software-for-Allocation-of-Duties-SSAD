import pandas as pd
from rapidfuzz import process, fuzz
import logging
import os
import warnings
from huggingface_hub import hf_hub_download

# Silence all the noise
logging.getLogger("llama_cpp").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

try:
    from llama_cpp import Llama
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

logging.basicConfig(level=logging.ERROR)

class RosterAnalyzer:
    def __init__(self):
        self.model_path = os.path.join(os.getcwd(), "models", "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
        self.llm = None
        # Added "staff" and "resource" to catch more header variations
        self.anchor_keywords = ["designation", "employee name", "sap id", "emp name", "name", "role", "s.no", "staff", "resource"]
        
        if AI_AVAILABLE:
            self._initialize_model()

    def _initialize_model(self):
        if not os.path.exists(self.model_path):
            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                hf_hub_download(repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF", filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf", local_dir=os.path.dirname(self.model_path))
            except Exception as e:
                print(f"Download failed: {e}")
                return
        try:
            self.llm = Llama(model_path=self.model_path, n_ctx=4096, n_threads=4, verbose=False)
            print("AI Engine Online.")
        except Exception as e:
            print(f"Failed to load AI: {e}")

    def _ask_ai_for_column(self, headers, goal, avoid_terms=[]):
        if not self.llm: return None
        header_list = [str(h).strip() for h in headers if "unnamed" not in str(h).lower()]
        avoid_instr = f"Do NOT select columns like {avoid_terms}." if avoid_terms else ""
        
        # Simplified prompt to reduce hallucinations
        prompt = f"""<|start_header_id|>system<|end_header_id|>
Select the EXACT column name from the list that matches: "{goal}".
List: {header_list}
{avoid_instr}
Reply ONLY with the column name. If None, reply "NONE".<|eot_id|><|start_header_id|>user<|end_header_id|>
Column name?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        try:
            output = self.llm(prompt, max_tokens=30, stop=["<|eot_id|>", "\n"], echo=False)
            answer = output['choices'][0]['text'].strip().replace('"', '').replace("'", "")
            # Validate answer exists in headers
            match = process.extractOne(answer, headers, scorer=fuzz.WRatio)
            if match and match[1] > 90: return match[0]
        except: pass
        return None

    def _find_column_keyword(self, headers, keyword):
        """Fallback: Math logic to find a column containing a keyword."""
        for col in headers:
            if keyword.lower() in str(col).lower():
                return col
        return None

    def _sanitize_dataframe(self, df):
        best_idx = -1
        for i, row in df.head(20).iterrows():
            row_str = " ".join([str(x).lower() for x in row.values])
            if any(k in row_str for k in self.anchor_keywords):
                print(f"DEBUG: Found Header at Row {i} -> {row.values[:4]}")
                best_idx = i
                break
        if best_idx != -1:
            df.columns = df.iloc[best_idx]
            df = df.iloc[best_idx + 1:].reset_index(drop=True)
        return df

    def get_shift_results(self, filepath, target_date_str, shift_type):
        final_results = []
        target_date = pd.to_datetime(target_date_str).date()
        date_short = target_date.strftime("%d-%b").lower()
        if date_short.startswith("0"): date_short_no_zero = date_short[1:]
        else: date_short_no_zero = date_short

        try:
            with pd.ExcelFile(filepath) as xl:
                for sheet_name in xl.sheet_names:
                    try:
                        df = xl.parse(sheet_name, header=None)
                        if df.empty: continue
                        
                        df = self._sanitize_dataframe(df)
                        headers = df.columns.tolist()

                        # 1. AI: Find Name Column
                        name_col = self._ask_ai_for_column(headers, "Employee Name", avoid_terms=["ID", "SAP", "No"])
                        # Fallback for Name
                        if not name_col:
                            name_col = self._find_column_keyword(headers, "name")
                        
                        if not name_col: 
                            print(f"DEBUG: Skipping {sheet_name} (No Name Column Found)")
                            continue

                        print(f"DEBUG: Sheet '{sheet_name}' | Name Col: {name_col}")

                        # 2. Determine Shape
                        date_col = None
                        is_wide_format = False

                        # Check WIDE (Headers = Dates)
                        for col in headers:
                            c_str = str(col).strip().lower()
                            if date_short in c_str or date_short_no_zero in c_str:
                                date_col = col; is_wide_format = True; break
                            try:
                                if pd.to_datetime(col, errors='coerce').date() == target_date:
                                    date_col = col; is_wide_format = True; break
                            except: pass
                        
                        # 3. Extract Matches
                        if is_wide_format and date_col:
                            print(f"DEBUG: Detected WIDE format. Date Col: {date_col}")
                            matches = df[df[date_col].astype(str).fillna('').str.strip().str.upper() == shift_type.upper()]
                            if not matches.empty:
                                names = matches[name_col].dropna().astype(str).tolist()
                                final_results.append({"sheet": sheet_name, "matches": [n for n in names if len(str(n)) > 1]})
                        
                        else:
                            # --- LONG LOGIC (The Fix) ---
                            print(f"DEBUG: Checking LONG format...")
                            
                            # Try AI first
                            ai_date_col = self._ask_ai_for_column(headers, "The Date column")
                            ai_shift_col = self._ask_ai_for_column(headers, "The Shift Code column")
                            
                            # Fallback: If AI missed "Duty Date", use keyword matching
                            if not ai_date_col: ai_date_col = self._find_column_keyword(headers, "date")
                            if not ai_date_col: ai_date_col = self._find_column_keyword(headers, "time") # timeline
                            
                            if not ai_shift_col: ai_shift_col = self._find_column_keyword(headers, "shift")
                            if not ai_shift_col: ai_shift_col = self._find_column_keyword(headers, "allocation") # allocation
                            if not ai_shift_col: ai_shift_col = self._find_column_keyword(headers, "code")

                            print(f"DEBUG: Long Format decision -> DateCol: {ai_date_col}, ShiftCol: {ai_shift_col}")

                            if ai_date_col and ai_shift_col:
                                # Normalize user shift (remove .0 for floats)
                                target_shift = str(shift_type).strip().upper()
                                
                                # Filter by Date
                                df_date = df[pd.to_datetime(df[ai_date_col], errors='coerce').dt.date == target_date]
                                print(f"DEBUG: Rows matching date {target_date}: {len(df_date)}")
                                
                                # Filter by Shift (Robust string matching)
                                matches = df_date[df_date[ai_shift_col].astype(str).fillna('').str.strip().str.upper() == target_shift]
                                print(f"DEBUG: Rows matching shift {target_shift}: {len(matches)}")

                                if not matches.empty:
                                    names = matches[name_col].dropna().astype(str).tolist()
                                    final_results.append({"sheet": sheet_name, "matches": [n for n in names if len(str(n)) > 1]})

                    except Exception as e:
                        print(f"Sheet Error: {e}")
                        continue     
        except Exception as e:
            raise e
            
        return final_results