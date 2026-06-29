import streamlit as st
from query_pipeline import answer_query
import config

st.set_page_config(page_title="music-rag", page_icon="🎵")
st.title("🎵 music-rag")
st.caption("SoundQuest の音楽理論記事を根拠に質問へ答える RAG デモ")

with st.sidebar:
    top_k = st.slider("取得チャンク数 (top_k)", 1, 10, config.TOP_K)
    songle_url = st.text_input("楽曲URL（任意・Songle解析済み）", "")

query = st.text_input("質問を入力", placeholder="例: ドミナントモーションとは？")

if st.button("質問する", type="primary") and query.strip():
    with st.spinner("検索・生成中…（初回はモデル読込で時間がかかります）"):
        result = answer_query(query.strip(), top_k=top_k,
                              songle_url=songle_url.strip() or None)
    st.markdown("### 回答")
    st.write(result["answer"])
    st.markdown("### 出典")
    for s in dict.fromkeys(result["sources"]):
        st.markdown(f"- `{s}`")
    with st.expander("取得チャンク（デバッグ）"):
        for i, c in enumerate(result["contexts"], 1):
            st.markdown(f"**{i}. {c['source']}**（score {c['score']:.3f}）")
            st.text(c["text"][:200])