import streamlit as st
from PIL import Image
from attack_algorithm import run_attack, get_sc_img
import io

st.title("Non-Target Boundary Attack GUI")

# 1. 画像アップロード
uploaded_file = st.file_uploader("Upload an image to attack", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    original_image_pil = Image.open(uploaded_file).resize((224, 224))

    # --- 設定エリア ---
    st.header("Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image_pil, caption="Original Image Preview", use_container_width=True)
    
    with col2:
        steps = st.number_input("Number of Steps", min_value=1, max_value=5000, value=2000)
        color = st.selectbox("Select Initial Image", ("Blue", "Red", "Green", "Noise", "Mean"))

        # Start Attackボタンを定義
        start_button = st.button("Start Attack")

    # --- 結果表示エリア ---
    # ボタンが押されたら、設定エリアの下に結果を表示
    if start_button:
        st.header("Attack Progress / Result")
        progress_bar = st.progress(0, text="Starting...")
        
        # 結果表示用に3つのカラムを作成
        res_col1, res_col2, res_col3 = st.columns(3)

        # 左のカラム：Initial Image（固定）
        # get_sc_imgに元画像を渡すように変更
        target_image_pil = get_sc_img(color, original_image_pil)
        res_col1.image(target_image_pil, caption="Initial Image", use_container_width=True)

        # 中央のカラム：Adversarial Image（動的更新用）
        adversarial_placeholder = res_col2.empty()
        
        # 右のカラム：Target Image（固定）
        res_col3.image(original_image_pil, caption="Target Image", use_container_width=True)

        # 攻撃の実行
        final_image = run_attack(original_image_pil, steps, color, progress_bar, adversarial_placeholder)

        if final_image:
            # PILイメージをバイトに変換
            buf = io.BytesIO()
            final_image.save(buf, format="PNG")
            byte_im = buf.getvalue()

            # 中央のカラムにダウンロードボタンを追加
            res_col2.download_button(
                label="Download Image",
                data=byte_im,
                file_name="adversarial_image.png",
                mime="image/png"
            )