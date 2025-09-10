import streamlit as st
from PIL import Image
from attack_algorithm import run_blue_attack 

# --- Session State ---
# 'attack_running'：攻撃の実行状態を管理
if 'attack_running' not in st.session_state:
    st.session_state.attack_running = False
# 'last_image'：最後に表示した画像を記憶するための新しい変数
if 'last_image' not in st.session_state:
    st.session_state.last_image = None

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
    btn_col1, btn_col2, _ = st.columns([1, 1, 5])
    
    with btn_col1:
        if st.button("Start Attack"):
            # 攻撃開始時に、前回の結果をリセット
            st.session_state.last_image = None
            st.session_state.attack_running = True
            # st.rerun() を追加して、UIを即座に更新
            st.rerun()

    with btn_col2:
        if st.button("Stop Attack"):
            st.session_state.attack_running = False
            st.warning("Attack stopping...")
            # st.rerun() を追加して、UIを即座に更新
            st.rerun()

    # 4. 結果表示エリア
    st.header("Attack Progress / Result")
    result_placeholder = st.empty()

    # 5. 攻撃の実行ロジック
    if st.session_state.attack_running:
        progress_bar = st.progress(0, text="Starting...")
        
        # run_blue_attackを呼び出し、結果をlast_imageに保存
        final_image = run_blue_attack(original_image_pil, steps, progress_bar, result_placeholder)
        st.session_state.last_image = final_image
        
        # 攻撃が終了したら、状態をFalseに
        st.session_state.attack_running = False
        # 結果を確定表示するために再実行
        st.rerun()

    # 6. 最後に生成された画像があれば、それを表示し続ける
    elif st.session_state.last_image is not None:
        result_placeholder.image(st.session_state.last_image, caption="Final Adversarial Image", use_container_width=True)

