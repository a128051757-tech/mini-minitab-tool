import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import itertools

# 設定頁面資訊
st.set_page_config(page_title="Mini Minitab (Web版)", page_icon="📊", layout="wide")

# 解決 matplotlib 中文顯示問題 (針對雲端環境建議用英文或安裝字型，這裡先用英文介面)
plt.style.use('ggplot')

def main():
    st.title("🏭 Process Engineer's Mini-Tool")
    st.markdown("### 製程工程師專用 - 免費 Minitab 替代方案")

    # 側邊欄選單
    menu = ["🏠 首頁", "📈 製程能力 (Cpk)", "📊 繪圖分析 (Plots)", "🧪 實驗設計 (DOE)"]
    choice = st.sidebar.selectbox("請選擇功能", menu)

    if choice == "🏠 首頁":
        st.info("歡迎使用！請從左側選單選擇您需要的功能。")
        st.write("目前支援功能：")
        st.write("- **Cpk Analysis**: 支援 CSV 上傳或自動生成模擬數據。")
        st.write("- **Plots**: 箱型圖、散佈圖、柏拉圖。")
        st.write("- **DOE**: 建立 2水準全因子實驗計畫表。")

    elif choice == "📈 製程能力 (Cpk)":
        st.header("Process Capability Analysis (Cpk)")
        
        # 資料來源選擇
        data_source = st.radio("選擇資料來源", ["使用模擬數據", "上傳 CSV 檔案"])
        
        data = []
        if data_source == "上傳 CSV 檔案":
            uploaded_file = st.file_uploader("請上傳 CSV 檔案 (需包含標題列)", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write("預覽資料:", df.head())
                col_name = st.selectbox("請選擇要分析的欄位 (數值)", df.select_dtypes(include=np.number).columns)
                data = df[col_name].dropna()
        else:
            # 產生模擬數據
            mean_input = st.number_input("設定模擬平均值", value=10.0)
            std_input = st.number_input("設定模擬標準差", value=0.1)
            data = np.random.normal(mean_input, std_input, 100)
            st.success(f"已生成 100 筆常態分佈數據 (Mean={mean_input}, Std={std_input})")

        if len(data) > 0:
            col1, col2 = st.columns(2)
            with col1:
                USL = st.number_input("規格上限 (USL)", value=float(np.mean(data) + 4*np.std(data)))
            with col2:
                LSL = st.number_input("規格下限 (LSL)", value=float(np.mean(data) - 4*np.std(data)))

            if st.button("計算 Cpk"):
                mean = np.mean(data)
                sigma = np.std(data, ddof=1)
                Cp = (USL - LSL) / (6 * sigma)
                Cpu = (USL - mean) / (3 * sigma)
                Cpl = (mean - LSL) / (3 * sigma)
                Cpk = min(Cpu, Cpl)

                st.metric(label="Cpk", value=f"{Cpk:.4f}", delta=f"Cp: {Cp:.4f}")
                st.write(f"Mean: {mean:.4f} | Sigma: {sigma:.4f}")

                # 繪圖
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(data, kde=True, color='green', stat='density', ax=ax, label='Data')
                
                # 畫規格線
                ax.axvline(USL, color='red', linestyle='--', linewidth=2, label='USL')
                ax.axvline(LSL, color='red', linestyle='--', linewidth=2, label='LSL')
                ax.set_title(f"Capability Histogram (Cpk={Cpk:.2f})")
                ax.legend()
                
                st.pyplot(fig)

    elif choice == "📊 繪圖分析 (Plots)":
        st.header("工程圖表繪製")
        plot_type = st.selectbox("選擇圖表類型", ["箱型圖 (Boxplot)", "柏拉圖 (Pareto)", "散佈圖 (Scatter)"])
        
        if plot_type == "箱型圖 (Boxplot)":
            st.subheader("多群組比較")
            # 這裡簡單起見，直接生成模擬數據演示
            if st.checkbox("使用範例數據演示"):
                data_a = np.random.normal(10.0, 0.2, 50)
                data_b = np.random.normal(10.2, 0.5, 50)
                df_box = pd.DataFrame({
                    'Value': np.concatenate([data_a, data_b]),
                    'Group': ['Machine A']*50 + ['Machine B']*50
                })
                fig = plt.figure(figsize=(8, 5))
                sns.boxplot(x='Group', y='Value', data=df_box, palette="Set2")
                st.pyplot(fig)
            else:
                st.info("請上傳含有類別與數值欄位的 CSV")

        elif plot_type == "柏拉圖 (Pareto)":
            st.subheader("不良原因分析")
            # 這裡簡單演示
            data = {'Defect': ['Scratch', 'Dimension', 'Burr', 'Short', 'Other'],
                    'Count': [150, 80, 40, 20, 10]}
            df_pareto = pd.DataFrame(data)
            
            # 使用者可以修改數據
            edited_df = st.data_editor(df_pareto, num_rows="dynamic")
            
            if st.button("繪製柏拉圖"):
                df_sorted = edited_df.sort_values(by='Count', ascending=False)
                df_sorted['Cum_Percent'] = df_sorted['Count'].cumsum() / df_sorted['Count'].sum() * 100
                
                fig, ax1 = plt.subplots()
                ax1.bar(df_sorted['Defect'], df_sorted['Count'], color='steelblue')
                ax1.set_ylabel('Count')
                
                ax2 = ax1.twinx()
                ax2.plot(df_sorted['Defect'], df_sorted['Cum_Percent'], color='red', marker='D')
                ax2.set_ylim(0, 110)
                ax2.set_ylabel('Cumulative %')
                
                st.pyplot(fig)

    elif choice == "🧪 實驗設計 (DOE)":
        st.header("Design of Experiments (DOE)")
        st.subheader("建立 2水準全因子設計")
        
        num_factors = st.number_input("因子數量 (Factors)", min_value=2, max_value=5, value=3)
        factor_names = []
        for i in range(num_factors):
            factor_names.append(st.text_input(f"因子 {i+1} 名稱", value=f"Factor_{chr(65+i)}"))
            
        if st.button("生成實驗計畫表"):
            levels = [-1, 1]
            design = list(itertools.product(levels, repeat=num_factors))
            df_doe = pd.DataFrame(design, columns=factor_names)
            # 增加一個空欄位讓使用者填結果
            df_doe['Response (Y)'] = ""
            
            st.write("### 您的實驗計畫矩陣")
            st.dataframe(df_doe)
            
            # 讓使用者下載 CSV
            csv = df_doe.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 下載實驗計畫 CSV",
                data=csv,
                file_name="doe_design.csv",
                mime="text/csv",
            )
            st.info("提示：下載後填入實驗結果 (Y)，未來可增加上傳分析功能。")

if __name__ == "__main__":
    main()