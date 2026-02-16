import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import itertools
import warnings

# --- 系統設定 ---
st.set_page_config(page_title="Process Engineer's Mini-Tool", page_icon="🏭", layout="wide")

# 移除舊版設定，改用標準寫法
# st.set_option('deprecation.showPyplotGlobalUse', False) <--- 這行已刪除

warnings.filterwarnings('ignore')

# 設定繪圖風格 (使用英文標籤以避免雲端環境中文亂碼)
plt.style.use('ggplot')

# ==========================================
# 主程式邏輯
# ==========================================
def main():
    st.sidebar.title("🏭 工程數據分析工具")
    st.sidebar.markdown("Process Engineer's Mini-Tool")
    
    menu = [
        "🏠 首頁 (Home)", 
        "📈 製程能力 (Cpk Analysis)", 
        "📊 統計繪圖 (EDA Plots)", 
        "🧪 假設檢定 (Hypothesis Testing)",
        "⚗️ 實驗設計 (DOE)",
        "📏 量測系統分析 (MSA)"
    ]
    
    choice = st.sidebar.radio("請選擇功能模組", menu)

    # -------------------------------------------------------------------------
    # 0. 首頁
    # -------------------------------------------------------------------------
    if choice == "🏠 首頁 (Home)":
        st.title("歡迎使用製程工程師數據分析工具")
        st.markdown("""
        這是一個基於 Python 與 Streamlit 開發的 web 應用程式，旨在提供類似 Minitab 的核心功能，
        協助製程工程師 (PE) 快速進行數據分析。

        ### 目前支援功能：
        1.  **Cpk Analysis**: 製程能力分析 (包含 Histogram 與常態曲線)。
        2.  **EDA Plots**: 箱型圖 (Boxplot)、柏拉圖 (Pareto)、散佈圖 (Scatter)。
        3.  **Hypothesis Testing**: T檢定 (t-test)、變異數分析 (ANOVA)。
        4.  **DOE**: 建立全因子實驗計畫 (Full Factorial Design)。
        5.  **MSA**: 
            * Type 1 Gage Study (Cgk)
            * Gage Linearity (線性度)
            * Gage R&R (ANOVA法)
            * Gage Stability (穩定性 Xbar-R)
        
        ---
        **使用說明：** 請從左側選單選擇功能，並依照提示上傳 CSV 檔案或輸入參數。
        """)

    # -------------------------------------------------------------------------
    # 1. 製程能力 (Cpk)
    # -------------------------------------------------------------------------
    elif choice == "📈 製程能力 (Cpk Analysis)":
        st.header("Process Capability Analysis (Cpk/Ppk)")
        
        data_source = st.radio("資料來源", ["模擬數據 (Demo)", "上傳 CSV"])
        
        data = []
        if data_source == "上傳 CSV":
            uploaded_file = st.file_uploader("上傳 CSV (需含標題列)", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                col = st.selectbox("選擇分析欄位", df.select_dtypes(include=np.number).columns)
                data = df[col].dropna().values
        else:
            mean_sim = st.number_input("模擬平均值", value=10.0)
            std_sim = st.number_input("模擬標準差", value=0.1)
            data = np.random.normal(mean_sim, std_sim, 100)
            st.info(f"已生成 100 筆模擬數據 (Mean={mean_sim})")

        if len(data) > 0:
            c1, c2 = st.columns(2)
            # 避免全 0 數據導致計算錯誤
            current_mean = float(np.mean(data)) if len(data) > 0 else 0.0
            current_std = float(np.std(data)) if len(data) > 0 else 1.0
            
            usl = c1.number_input("USL (規格上限)", value=current_mean + 4*current_std)
            lsl = c2.number_input("LSL (規格下限)", value=current_mean - 4*current_std)

            if st.button("計算 Cpk"):
                mean = np.mean(data)
                std = np.std(data, ddof=1)
                
                Cp = (usl - lsl) / (6 * std) if std != 0 else 0
                Cpu = (usl - mean) / (3 * std) if std != 0 else 0
                Cpl = (mean - lsl) / (3 * std) if std != 0 else 0
                Cpk = min(Cpu, Cpl)
                
                st.metric("Cpk", f"{Cpk:.4f}", f"Cp: {Cp:.4f}")
                st.write(f"Mean: {mean:.4f}, Std Dev: {std:.4f}")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(data, kde=True, color='green', stat='density', ax=ax, label='Data')
                ax.axvline(usl, color='r', linestyle='--', label='USL')
                ax.axvline(lsl, color='r', linestyle='--', label='LSL')
                
                # 畫常態分佈線
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, mean, std)
                ax.plot(x, p, 'k', linewidth=2, label='Normal Dist')
                
                ax.legend()
                ax.set_title(f"Capability Analysis (Cpk={Cpk:.2f})")
                st.pyplot(fig)

    # -------------------------------------------------------------------------
    # 2. 統計繪圖 (Plots)
    # -------------------------------------------------------------------------
    elif choice == "📊 統計繪圖 (EDA Plots)":
        st.header("Exploratory Data Analysis Plots")
        plot_type = st.selectbox("選擇圖表", ["箱型圖 (Boxplot)", "柏拉圖 (Pareto)", "散佈圖 (Scatter)"])

        if plot_type == "箱型圖 (Boxplot)":
            st.info("比較不同群組的數據分佈 (例如：不同機台、不同模穴)。")
            # 簡易模擬
            if st.checkbox("使用模擬數據", value=True):
                d1 = np.random.normal(10, 0.2, 50)
                d2 = np.random.normal(10.2, 0.5, 50)
                df = pd.DataFrame({'Value': np.concatenate([d1,d2]), 'Group': ['A']*50 + ['B']*50})
                fig = plt.figure(figsize=(8,5))
                sns.boxplot(x='Group', y='Value', data=df)
                st.pyplot(fig)
            else:
                st.warning("請自行擴充上傳功能")

        elif plot_type == "柏拉圖 (Pareto)":
            st.info("80/20 法則分析 (不良原因排序)。")
            data = {'Defect': ['Scratch', 'Dimension', 'Burr', 'Short', 'Other'], 'Count': [150, 80, 40, 20, 10]}
            df = pd.DataFrame(data)
            edited_df = st.data_editor(df, num_rows="dynamic")
            
            if st.button("繪圖"):
                df_sorted = edited_df.sort_values(by='Count', ascending=False)
                df_sorted['Cum%'] = df_sorted['Count'].cumsum() / df_sorted['Count'].sum() * 100
                
                fig, ax1 = plt.subplots()
                ax1.bar(df_sorted['Defect'], df_sorted['Count'], color='steelblue')
                ax2 = ax1.twinx()
                ax2.plot(df_sorted['Defect'], df_sorted['Cum%'], color='red', marker='D')
                ax2.set_ylim(0, 110)
                ax2.axhline(80, color='gray', linestyle='--')
                st.pyplot(fig)

        elif plot_type == "散佈圖 (Scatter)":
            st.info("分析兩個變數之間的相關性 (例如：溫度 vs 尺寸)。")
            x = np.random.uniform(100, 200, 50)
            y = 0.5 * x + np.random.normal(0, 5, 50)
            df = pd.DataFrame({'Temp': x, 'Size': y})
            fig = plt.figure()
            sns.regplot(x='Temp', y='Size', data=df)
            st.pyplot(fig)

    # -------------------------------------------------------------------------
    # 3. 假設檢定 (Hypothesis)
    # -------------------------------------------------------------------------
    elif choice == "🧪 假設檢定 (Hypothesis Testing)":
        st.header("Hypothesis Testing")
        h_type = st.selectbox("檢定類型", ["雙樣本 T 檢定 (2-Sample t-test)", "單因子變異數分析 (One-Way ANOVA)"])

        if h_type == "雙樣本 T 檢定 (2-Sample t-test)":
            st.subheader("比較兩組平均值")
            c1, c2 = st.columns(2)
            t1 = c1.text_area("數據 A (逗號分隔)", "10.1, 10.2, 10.5, 9.9")
            t2 = c2.text_area("數據 B (逗號分隔)", "10.8, 10.9, 10.7, 10.6")
            
            if st.button("執行 T 檢定"):
                try:
                    a = [float(x) for x in t1.split(',')]
                    b = [float(x) for x in t2.split(',')]
                    t_stat, p = stats.ttest_ind(a, b, equal_var=False)
                    st.write(f"**P-Value**: {p:.4f}")
                    if p < 0.05:
                        st.error("Reject H0: 兩組有顯著差異")
                    else:
                        st.success("Fail to Reject H0: 兩組無顯著差異")
                except:
                    st.error("數據格式錯誤")

    # -------------------------------------------------------------------------
    # 4. 實驗設計 (DOE)
    # -------------------------------------------------------------------------
    elif choice == "⚗️ 實驗設計 (DOE)":
        st.header("Design of Experiments (DOE)")
        doe_mode = st.radio("模式", ["建立 2水準全因子", "建立 一般全因子 (多水準)"])

        if doe_mode == "建立 2水準全因子":
            factors = st.number_input("因子數量", 2, 5, 3)
            names = [st.text_input(f"因子 {i+1}", f"F{i+1}") for i in range(factors)]
            if st.button("生成設計表"):
                df = pd.DataFrame(list(itertools.product([-1, 1], repeat=factors)), columns=names)
                st.dataframe(df)
                st.download_button("下載 CSV", df.to_csv(index=False), "doe_design.csv")

        else:
            factors_num = st.number_input("因子數量", 1, 5, 2)
            levels_dict = {}
            for i in range(factors_num):
                fname = st.text_input(f"因子 {i+1} 名稱", f"Factor_{chr(65+i)}")
                lvl_str = st.text_input(f"{fname} 水準 (逗號分隔)", "100, 200, 300")
                levels_dict[fname] = [x.strip() for x in lvl_str.split(',')]
            
            if st.button("生成多水準設計表"):
                keys, values = zip(*levels_dict.items())
                df = pd.DataFrame(list(itertools.product(*values)), columns=keys)
                st.dataframe(df)

    # -------------------------------------------------------------------------
    # 5. 量測系統分析 (MSA)
    # -------------------------------------------------------------------------
    elif choice == "📏 量測系統分析 (MSA)":
        st.header("Measurement System Analysis (MSA)")
        msa_type = st.selectbox("選擇 MSA 類型", 
            ["1. Type 1 Gage Study (Cgk)", 
             "2. Gage Linearity (線性度)", 
             "3. Gage R&R (ANOVA)", 
             "4. Gage Stability (穩定性)"])

        # --- 5.1 Type 1 Gage Study ---
        if msa_type == "1. Type 1 Gage Study (Cgk)":
            st.subheader("Type 1 Gage Study")
            st.info("評估量具的重複性(Cg)與偏誤(Cgk)。需單一標準件量測 >=25 次。")
            
            c1, c2, c3 = st.columns(3)
            ref = c1.number_input("參考值 (Ref)", 10.0)
            tol = c2.number_input("公差帶 (Tolerance)", 2.0)
            pct = c3.number_input("Cg要求 % (預設 20%)", 20.0) / 100.0

            file = st.file_uploader("上傳 CSV (單一數值欄位)", type="csv", key="cgk")
            if file:
                df = pd.read_csv(file)
                col = st.selectbox("數值欄位", df.select_dtypes(include=np.number).columns)
                data = df[col].dropna().values
                
                if len(data) < 1: st.stop()
                
                mean = np.mean(data)
                std = np.std(data, ddof=1)
                bias = mean - ref
                K = pct * tol
                Cg = K / (6 * std) if std != 0 else 0
                Cgk = (K / 2 - abs(bias)) / (3 * std) if std != 0 else 0
                
                st.write(f"**Bias**: {bias:.4f}, **StDev**: {std:.4f}")
                c1, c2 = st.columns(2)
                c1.metric("Cg", f"{Cg:.2f}")
                c2.metric("Cgk", f"{Cgk:.2f}")
                
                if Cgk > 1.33: st.success("✅ Cgk 合格")
                else: st.error("❌ Cgk 不合格")
                
                fig, ax = plt.subplots()
                ax.plot(data, 'o-')
                ax.axhline(ref, color='g', label='Ref')
                ax.axhline(ref + 0.1*tol, color='r', linestyle='--', label='Limit')
                ax.axhline(ref - 0.1*tol, color='r', linestyle='--')
                ax.set_title("Run Chart")
                st.pyplot(fig)

        # --- 5.2 Linearity ---
        elif msa_type == "2. Gage Linearity (線性度)":
            st.subheader("Gage Linearity & Bias")
            st.info("需欄位: 'Ref' (標準值), 'Value' (量測值)")
            
            file = st.file_uploader("上傳 CSV", type="csv", key="lin")
            if file:
                df = pd.read_csv(file)
                c1, c2 = st.columns(2)
                ref_col = c1.selectbox("標準值欄位", df.columns)
                val_col = c2.selectbox("量測值欄位", df.columns)
                
                df['Bias'] = df[val_col] - df[ref_col]
                
                # 迴歸分析 Bias = a + b * Ref
                X = sm.add_constant(df[ref_col])
                model = sm.OLS(df['Bias'], X).fit()
                
                st.write(f"**方程式**: Bias = {model.params['const']:.4f} + {model.params[ref_col]:.4f} * Ref")
                st.write(f"**Slope P-Value**: {model.pvalues[ref_col]:.4f}")
                
                if model.pvalues[ref_col] < 0.05:
                    st.error("線性度不佳 (Bias 隨尺寸變化)")
                else:
                    st.success("線性度良好 (Bias 穩定)")
                
                fig, ax = plt.subplots()
                sns.regplot(x=ref_col, y='Bias', data=df, ax=ax)
                st.pyplot(fig)

        # --- 5.3 Gage R&R (ANOVA) ---
        elif msa_type == "3. Gage R&R (ANOVA)":
            st.subheader("Gage R&R (Crossed ANOVA)")
            st.info("需欄位: 'Part', 'Operator', 'Value'")
            
            file = st.file_uploader("上傳 CSV", type="csv", key="grr")
            if file:
                df = pd.read_csv(file)
                c1, c2, c3 = st.columns(3)
                p_col = c1.selectbox("Part", df.columns)
                o_col = c2.selectbox("Operator", df.columns)
                v_col = c3.selectbox("Value", df.select_dtypes(include=np.number).columns)
                
                if st.button("執行 ANOVA"):
                    try:
                        # 轉為分類變數
                        df[p_col] = df[p_col].astype(str)
                        df[o_col] = df[o_col].astype(str)
                        
                        formula = f"{v_col} ~ C({p_col}) + C({o_col}) + C({p_col}):C({o_col})"
                        model = ols(formula, data=df).fit()
                        aov_table = anova_lm(model, typ=2)
                        
                        # 變異數成分估算 (簡化版)
                        ms_part = aov_table.loc[f"C({p_col})", 'mean_sq']
                        ms_oper = aov_table.loc[f"C({o_col})", 'mean_sq']
                        ms_inter = aov_table.loc[f"C({p_col}):C({o_col})", 'mean_sq']
                        ms_error = aov_table.loc['Residual', 'mean_sq']
                        
                        n_p = df[p_col].nunique()
                        n_o = df[o_col].nunique()
                        n_rep = len(df) / (n_p * n_o)
                        
                        var_repeat = ms_error
                        var_inter = max(0, (ms_inter - ms_error) / n_rep)
                        var_repro = max(0, (ms_oper - ms_inter) / (n_p * n_rep)) + var_inter
                        var_part = max(0, (ms_part - ms_inter) / (n_o * n_rep))
                        
                        var_grr = var_repeat + var_repro
                        var_total = var_grr + var_part
                        
                        pct_study_var = (np.sqrt(var_grr) / np.sqrt(var_total)) * 100
                        
                        st.metric("% GRR (Study Var)", f"{pct_study_var:.2f}%")
                        if pct_study_var < 10: st.success("🟢 優秀 (<10%)")
                        elif pct_study_var < 30: st.warning("🟡 可接受 (10-30%)")
                        else: st.error("🔴 不合格 (>30%)")
                        
                        # 繪圖
                        fig, ax = plt.subplots()
                        sns.pointplot(x=p_col, y=v_col, hue=o_col, data=df, ax=ax)
                        ax.set_title("Operator * Part Interaction")
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"計算錯誤: {e}")

        # --- 5.4 Gage Stability ---
        elif msa_type == "4. Gage Stability (穩定性)":
            st.subheader("Gage Stability (Xbar-R Chart)")
            st.info("需欄位: 'Value', 'Group'(日期/組別)")
            
            file = st.file_uploader("上傳 CSV", type="csv", key="stab")
            if file:
                df = pd.read_csv(file)
                c1, c2 = st.columns(2)
                v_col = c1.selectbox("數值欄位", df.select_dtypes(include=np.number).columns)
                g_col = c2.selectbox("分組欄位", df.columns)
                
                if st.button("執行穩定性分析"):
                    try:
                        df[g_col] = df[g_col].astype(str)
                        grouped = df.groupby(g_col)[v_col].agg(['mean', 'min', 'max', 'count'])
                        grouped['range'] = grouped['max'] - grouped['min']
                        grouped = grouped.reset_index()
                        
                        n_val = grouped['count'].mean()
                        n = int(round(n_val))
                        st.write(f"平均樣本數 (n): {n}")
                        
                        # SPC Constants (n=2 to 10)
                        spc = {
                            2: {'A2': 1.880, 'D4': 3.267, 'd2': 1.128},
                            3: {'A2': 1.023, 'D4': 2.574, 'd2': 1.693},
                            4: {'A2': 0.729, 'D4': 2.282, 'd2': 2.059},
                            5: {'A2': 0.577, 'D4': 2.114, 'd2': 2.326},
                            6: {'A2': 0.483, 'D4': 2.004, 'd2': 2.534},
                            7: {'A2': 0.419, 'D4': 1.924, 'd2': 2.704},
                            8: {'A2': 0.373, 'D4': 1.864, 'd2': 2.847},
                            9: {'A2': 0.337, 'D4': 1.816, 'd2': 2.970},
                            10:{'A2': 0.308, 'D4': 1.777, 'd2': 3.078},
                        }
                        
                        if n in spc:
                            const = spc[n]
                            xb = grouped['mean'].mean()
                            rb = grouped['range'].mean()
                            
                            ucl_x = xb + const['A2'] * rb
                            lcl_x = xb - const['A2'] * rb
                            ucl_r = const['D4'] * rb
                            lcl_r = 0 
                            
                            st.write(f"Xbar Limit: [{lcl_x:.2f}, {ucl_x:.2f}], R Limit: [0, {ucl_r:.2f}]")
                            
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                            
                            # Xbar
                            ax1.plot(grouped[g_col], grouped['mean'], 'o-b')
                            ax1.axhline(xb, color='g')
                            ax1.axhline(ucl_x, color='r', linestyle='--')
                            ax1.axhline(lcl_x, color='r', linestyle='--')
                            ax1.set_title("Xbar Chart")
                            
                            # R
                            ax2.plot(grouped[g_col], grouped['range'], 'o-b')
                            ax2.axhline(rb, color='g')
                            ax2.axhline(ucl_r, color='r', linestyle='--')
                            ax2.set_title("R Chart")
                            
                            st.pyplot(fig)
                        else:
                            st.error(f"目前支援 n=2~10, 您的 n={n}")
                            
                    except Exception as e:
                        st.error(f"分析失敗: {e}")

if __name__ == "__main__":
    main()
