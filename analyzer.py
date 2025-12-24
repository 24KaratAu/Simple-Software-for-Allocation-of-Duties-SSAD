import pandas as pd
from rapidfuzz import process, fuzz
import logging
import os
import warnings
from huggingface_hub import hf_hub_download

# Silence logging
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
        # Anchors to find the header row
        self.anchor_keywords = ["designation", "employee name", "sap id", "emp name", "name", "role", "s.no", "staff", "resource"]
        
        self.schema_map = {
            "employee_name": ["employee name", "name", "staff", "personnel", "worker", "resource"],
            "date": ["date", "day", "timeline"], 
            "shift": ["shift", "duty", "assignment", "allocation"]
        }

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
        
        # We removed "ID" from the default avoidance list to allow "Resource ID"
        avoid_instr = f"Do NOT select columns like {avoid_terms}." if avoid_terms else ""
        
        prompt = f"""<|start_header_id|>system<|end_header_id|>
Select the EXACT column name from the list that matches: "{goal}".
List: {header_list}
{avoid_instr}
Reply ONLY with the column name. If None, reply "NONE".<|eot_id|><|start_header_id|>user<|end_header_id|>
Column name?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        try:
            output = self.llm(prompt, max_tokens=30, stop=["<|eot_id|>", "\n"], echo=False)
            answer = output['choices'][0]['text'].strip().replace('"', '').replace("'", "")
            match = process.extractOne(answer, headers, scorer=fuzz.WRatio)
            if match and match[1] > 90: return match[0]
        except: pass
        return None

    def _find_column_keyword(self, headers, keyword):
        """Math fallback."""
        for col in headers:
            if keyword.lower() in str(col).lower():
                return col
        return None

    def _is_valid_name_column(self, df, col_name):
        """Rejects columns that are mostly numbers (like SAP IDs)."""
        try:
            sample = df[col_name].dropna().astype(str).head(10).tolist()
            if not sample: return False
            # Count how many look like pure numbers
            digit_count = sum(1 for x in sample if x.replace('.','').isdigit())
            if digit_count > len(sample) * 0.5:
                print(f"DEBUG: Rejected '{col_name}' (Too many numbers: {sample[:3]})")
                return False
            return True
        except: return True

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

                        # 1. AI: Find Name Column (Allow IDs now, check validity later)
                        name_col = self._ask_ai_for_column(headers, "Employee Name", avoid_terms=["No", "S.No"])
                        
                        # Fallback for Name
                        if not name_col: name_col = self._find_column_keyword(headers, "name")
                        if not name_col: name_col = self._find_column_keyword(headers, "resource") # For Resource ID

                        # Validation Check
                        if name_col and not self._is_valid_name_column(df, name_col):
                            print(f"DEBUG: AI picked '{name_col}' but it failed validation.")
                            name_col = None 

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
                            # --- LONG LOGIC ---
                            print(f"DEBUG: Checking LONG format...")
                            
                            ai_date_col = self._ask_ai_for_column(headers, "The Date column")
                            ai_shift_col = self._ask_ai_for_column(headers, "The Shift Code column")
                            
                            # Fallback Logic
                            if not ai_date_col: ai_date_col = self._find_column_keyword(headers, "date")
                            if not ai_date_col: ai_date_col = self._find_column_keyword(headers, "time") # Timeline
                            
                            if not ai_shift_col: ai_shift_col = self._find_column_keyword(headers, "shift")
                            if not ai_shift_col: ai_shift_col = self._find_column_keyword(headers, "allocation") # Allocation
                            
                            print(f"DEBUG: Long Format decision -> DateCol: {ai_date_col}, ShiftCol: {ai_shift_col}")

                            if ai_date_col and ai_shift_col:
                                target_shift = str(shift_type).strip().upper()
                                
                                # Convert column to dates (Handle errors gracefully)
                                df['temp_date_parsed'] = pd.to_datetime(df[ai_date_col], errors='coerce').dt.date
                                
                                # DEBUG: Show what dates we actually found
                                print(f"DEBUG: First 3 dates found in '{ai_date_col}': {df['temp_date_parsed'].head(3).tolist()}")

                                df_date = df[df['temp_date_parsed'] == target_date]
                                print(f"DEBUG: Rows matching date {target_date}: {len(df_date)}")
                                
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