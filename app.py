import streamlit as st
from PIL import Image
from attack_algorithm import run_blue_attack 

st.title("Non-Target Boundary Attack GUI (Blue Image)")

# 1. 画像アップロード
uploaded_file = st.file_uploader("Upload an image to attack", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    original_image_pil = Image.open(uploaded_file).resize((224, 224))
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image_pil, caption="Original Image", use_container_width=True)
    with col2:
        # 2. ステップ数 設定
        steps = st.number_input("Number of Steps", min_value=1, max_value=5000, value=2000)

    # 3. 操作ボタン
    st.header("Controls")
    if st.button("Start Attack"):
        # 4. 結果表示エリア
        st.header("Attack Progress / Result")
        result_placeholder = st.empty()
        progress_bar = st.progress(0, text="Starting...")
        
        # 5. 攻撃の実行
        final_image = run_blue_attack(original_image_pil, steps, progress_bar, result_placeholder)
        
        # 最後に最終結果を確定表示
        result_placeholder.image(final_image, caption="Final Adversarial Image", use_container_width=True)

