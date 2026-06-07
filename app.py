"""
Sistema de Modelação Comportamental
Universidade Mandume Ya N'Demufayo — Engenharia Informática 2026
Autores: Angelina dos Santos · João Tchiweyengue · Josimar Carlos

Versão Streamlit — optimizada para deploy no Render
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
import data.database as db
import models.predictor as predictor

# ─── Configuração da página ───────────────────────────────────────
st.set_page_config(
    page_title="Modelação Comportamental",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personalizado ────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] .stRadio label { color: #cbd5e1; font-size: 15px; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    padding: 8px 12px; border-radius: 8px; margin-bottom: 4px;
    display: block; cursor: pointer;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: #1e293b;
}
.metric-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; padding: 20px 24px; margin-bottom: 12px;
}
.metric-value { font-size: 2rem; font-weight: 700; }
.metric-label { font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.section-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; }
.muted { color: #64748b; font-size: 0.9rem; }
div[data-testid="stExpander"] { background: #1e293b; border: 1px solid #334155; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Inicialização ────────────────────────────────────────────────
@st.cache_resource
def inicializar():
    db.init_db()
    db.adicionar_campo_resultado_real()
    db.popular_dados_demo()
    predictor.inicializar_modelo()
    return True

inicializar()

# ─── Paleta de cores ──────────────────────────────────────────────
AZUL    = "#3b82f6"
VERDE   = "#10b981"
AMBER   = "#f59e0b"
VERM    = "#ef4444"
ROXO    = "#8b5cf6"
CIANO   = "#06b6d4"
DARK_BG = "#1e293b"
DARKER  = "#0f172a"
TEXT    = "#f1f5f9"
MUTED   = "#64748b"
GRID    = "#334155"
PALETTE = [AZUL, VERDE, AMBER, VERM, ROXO, CIANO, "#f97316", "#ec4899"]


def _tema_fig(fig, axes=None):
    fig.patch.set_facecolor(DARK_BG)
    if axes is None:
        axes = fig.get_axes()
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(DARKER)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.5)


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎓 Modelação\nComportamental")
    st.markdown(f"<span class='muted'>{datetime.now().strftime('%d/%m/%Y')}</span>", unsafe_allow_html=True)
    st.markdown("---")

    pagina = st.radio(
        "Navegação",
        ["🏠  Dashboard", "👥  Estudantes", "📚  Sessões de Estudo",
         "📱  Hábitos Digitais", "🗓️  Rotina Diária",
         "📊  Gráficos", "🔮  Previsão IA", "⚙️  Painel ML"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("<span class='muted'>UMN · Eng. Informática 2026</span>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def selector_estudante(key="sel"):
    estudantes = db.listar_estudantes()
    if not estudantes:
        st.warning("Nenhum estudante registado. Vai a **Estudantes** e adiciona um.")
        return None, None
    opcoes = {f"{e['nome']} (id {e['id']})": e["id"] for e in estudantes}
    escolha = st.selectbox("Estudante", list(opcoes.keys()), key=key)
    return opcoes[escolha], db.obter_estudante(opcoes[escolha])


def gauge_color(val, low=0.3, high=0.6):
    if val < low: return VERDE
    if val < high: return AMBER
    return VERM


# ════════════════════════════════════════════════════════════════
# PÁGINA: DASHBOARD
# ════════════════════════════════════════════════════════════════

if pagina == "🏠  Dashboard":
    st.markdown("<div class='section-title'>Dashboard</div>", unsafe_allow_html=True)

    estudantes = db.listar_estudantes()
    total_est  = len(estudantes)
    all_notas, total_sess = [], 0
    for e in estudantes:
        sess = db.listar_sessoes(e["id"])
        total_sess += len(sess)
        all_notas.extend([s["nota"] for s in sess if s["nota"] is not None])

    media_geral = round(float(np.mean(all_notas)), 1) if all_notas else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("👥 Estudantes", total_est)
    with c2:
        st.metric("📚 Sessões Totais", total_sess)
    with c3:
        st.metric("📊 Média Geral", f"{media_geral} / 20")
    with c4:
        info = predictor.obter_info_modelo()
        estado = "✅ Pronto" if info["pronto"] else "⏳ A treinar…"
        st.metric("🤖 Modelo ML", estado)

    st.markdown("---")

    if estudantes:
        st.subheader("📈 Visão Geral — Todos os Estudantes")
        resumo_rows = []
        for e in estudantes:
            sess  = db.listar_sessoes(e["id"])
            hab   = db.listar_habitos(e["id"])
            rot   = db.listar_rotinas(e["id"])
            if sess or hab or rot:
                res = predictor.analisar_estudante(sess, hab, rot)
                resumo_rows.append({
                    "Nome": e["nome"],
                    "Curso": e["curso"] or "—",
                    "Ano": e["ano"] or "—",
                    "Nota Média": round(res["features"].get("media_nota", 0), 1),
                    "Risco Reprov.": f"{res['risco_reprovacao']*100:.0f}%",
                    "Risco Burnout": f"{res['risco_burnout']*100:.0f}%",
                    "Produtividade": f"{res['score_produtividade']:.0f}/100",
                    "Perfil": res["perfil"],
                })

        if resumo_rows:
            df = pd.DataFrame(resumo_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Gráfico de barras: produtividade por estudante
        if resumo_rows:
            st.subheader("🏆 Score de Produtividade por Estudante")
            nomes  = [r["Nome"].split()[0] for r in resumo_rows]
            scores = [float(r["Produtividade"].replace("/100","")) for r in resumo_rows]
            fig, ax = plt.subplots(figsize=(max(6, len(nomes)*1.2), 3.5))
            _tema_fig(fig, ax)
            bars = ax.bar(nomes, scores, color=PALETTE[:len(nomes)], width=0.5)
            ax.set_ylim(0, 105)
            ax.set_ylabel("Score")
            ax.set_title("Produtividade por Estudante", color=TEXT)
            for bar, s in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+2,
                        f"{s:.0f}", ha="center", va="bottom", color=TEXT, fontsize=9)
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Sem dados ainda. Vai a **Estudantes** para adicionar o primeiro estudante.")


# ════════════════════════════════════════════════════════════════
# PÁGINA: ESTUDANTES
# ════════════════════════════════════════════════════════════════

elif pagina == "👥  Estudantes":
    st.markdown("<div class='section-title'>👥 Estudantes</div>", unsafe_allow_html=True)

    tab_lista, tab_add, tab_remove = st.tabs(["Lista", "Adicionar", "Remover"])

    with tab_lista:
        estudantes = db.listar_estudantes()
        if estudantes:
            df = pd.DataFrame(estudantes).rename(columns={
                "id":"ID","nome":"Nome","idade":"Idade",
                "curso":"Curso","ano":"Ano","criado_em":"Criado em",
                "resultado_real":"Resultado","data_resultado":"Data Resultado",
            })
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum estudante registado.")

    with tab_add:
        with st.form("form_add_estudante"):
            nome   = st.text_input("Nome completo")
            c1, c2 = st.columns(2)
            with c1: idade = st.number_input("Idade", 14, 60, 20)
            with c2: ano   = st.number_input("Ano académico", 1, 6, 1)
            curso  = st.text_input("Curso", "Eng. Informática")
            if st.form_submit_button("➕ Adicionar Estudante", type="primary"):
                if nome.strip():
                    db.adicionar_estudante(nome.strip(), int(idade), curso.strip(), int(ano))
                    st.success(f"✅ **{nome}** adicionado com sucesso!")
                    st.rerun()
                else:
                    st.error("O nome é obrigatório.")

    with tab_remove:
        estudantes = db.listar_estudantes()
        if estudantes:
            opcoes = {f"{e['nome']} (id {e['id']})": e["id"] for e in estudantes}
            escolha = st.selectbox("Seleccionar estudante a remover", list(opcoes.keys()))
            st.warning("⚠️ Esta acção remove também todas as sessões, hábitos e rotinas associadas.")
            if st.button("🗑️ Remover Estudante", type="secondary"):
                db.remover_estudante(opcoes[escolha])
                st.success("Estudante removido.")
                st.rerun()
        else:
            st.info("Sem estudantes para remover.")


# ════════════════════════════════════════════════════════════════
# PÁGINA: SESSÕES DE ESTUDO
# ════════════════════════════════════════════════════════════════

elif pagina == "📚  Sessões de Estudo":
    st.markdown("<div class='section-title'>📚 Sessões de Estudo</div>", unsafe_allow_html=True)

    eid, estudante = selector_estudante("sel_sess")
    if eid:
        tab_lista, tab_add = st.tabs(["Histórico", "Registar Sessão"])

        with tab_lista:
            sessoes = db.listar_sessoes(eid)
            if sessoes:
                df = pd.DataFrame(sessoes).rename(columns={
                    "id":"ID","estudante_id":"Est.","data":"Data",
                    "duracao_min":"Duração (min)","materia":"Matéria",
                    "concentracao":"Concentração","nota":"Nota","observacoes":"Obs.",
                })
                st.dataframe(df.drop(columns=["Est."], errors="ignore"),
                             use_container_width=True, hide_index=True)
                st.caption(f"Total: {len(sessoes)} sessões registadas")
            else:
                st.info("Sem sessões registadas para este estudante.")

        with tab_add:
            materias = ["Algoritmos","BD","POO","Redes","Matemática","SO",
                        "Física","Inglês","Gestão","Outra"]
            with st.form("form_add_sessao"):
                data      = st.date_input("Data", datetime.today())
                c1, c2   = st.columns(2)
                with c1:
                    duracao = st.number_input("Duração (min)", 10, 480, 60)
                    materia = st.selectbox("Matéria", materias)
                with c2:
                    conc = st.slider("Concentração", 1, 10, 7)
                    nota = st.number_input("Nota (0–20)", 0.0, 20.0, 12.0, step=0.5)
                obs = st.text_area("Observações (opcional)", height=80)
                if st.form_submit_button("💾 Guardar Sessão", type="primary"):
                    db.adicionar_sessao(eid, data.strftime("%Y-%m-%d"),
                                        int(duracao), materia, int(conc), float(nota), obs)
                    st.success("✅ Sessão registada!")
                    st.rerun()


# ════════════════════════════════════════════════════════════════
# PÁGINA: HÁBITOS DIGITAIS
# ════════════════════════════════════════════════════════════════

elif pagina == "📱  Hábitos Digitais":
    st.markdown("<div class='section-title'>📱 Hábitos Digitais</div>", unsafe_allow_html=True)

    eid, estudante = selector_estudante("sel_hab")
    if eid:
        tab_lista, tab_add = st.tabs(["Histórico", "Registar Hábito"])

        with tab_lista:
            habitos = db.listar_habitos(eid)
            if habitos:
                df = pd.DataFrame(habitos)
                df = df.rename(columns={
                    "data":"Data","whatsapp_h":"WhatsApp(h)","youtube_h":"YouTube(h)",
                    "instagram_h":"Instagram(h)","jogos_h":"Jogos(h)",
                    "estudo_online_h":"Est.Online(h)","periodo_uso":"Período","sono_h":"Sono(h)",
                })
                st.dataframe(df.drop(columns=["id","estudante_id"], errors="ignore"),
                             use_container_width=True, hide_index=True)

                # Gráfico pizza de distrações
                st.subheader("Distribuição de Distrações Digitais")
                wh = np.mean([h["whatsapp_h"] for h in habitos])
                yh = np.mean([h["youtube_h"] for h in habitos])
                ig = np.mean([h["instagram_h"] for h in habitos])
                jg = np.mean([h["jogos_h"] for h in habitos])
                vals  = [wh, yh, ig, jg]
                lbls  = ["WhatsApp","YouTube","Instagram","Jogos"]
                fig, ax = plt.subplots(figsize=(5, 4))
                _tema_fig(fig, ax)
                wedges, texts, autotexts = ax.pie(
                    vals, labels=lbls, autopct="%1.0f%%",
                    colors=PALETTE[:4], startangle=90,
                    textprops={"color": TEXT, "fontsize": 9}
                )
                for at in autotexts: at.set_color(DARKER)
                ax.set_title("Médias de Distracção Digital", color=TEXT)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Sem hábitos registados.")

        with tab_add:
            with st.form("form_add_habito"):
                data = st.date_input("Data", datetime.today(), key="data_hab")
                c1, c2 = st.columns(2)
                with c1:
                    wh = st.number_input("WhatsApp (h)", 0.0, 12.0, 1.0, step=0.5)
                    yh = st.number_input("YouTube (h)", 0.0, 12.0, 1.0, step=0.5)
                    ig = st.number_input("Instagram (h)", 0.0, 12.0, 0.5, step=0.5)
                with c2:
                    jg = st.number_input("Jogos (h)", 0.0, 12.0, 0.0, step=0.5)
                    eo = st.number_input("Estudo Online (h)", 0.0, 12.0, 0.5, step=0.5)
                    sono = st.number_input("Sono (h)", 0.0, 12.0, 7.0, step=0.5)
                periodo = st.radio("Período principal de uso", ["manha","tarde","noite"], horizontal=True)
                if st.form_submit_button("💾 Guardar Hábito", type="primary"):
                    db.adicionar_habito(eid, data.strftime("%Y-%m-%d"),
                                        wh, yh, ig, jg, eo, periodo, sono)
                    st.success("✅ Hábito digital registado!")
                    st.rerun()


# ════════════════════════════════════════════════════════════════
# PÁGINA: ROTINA DIÁRIA
# ════════════════════════════════════════════════════════════════

elif pagina == "🗓️  Rotina Diária":
    st.markdown("<div class='section-title'>🗓️ Rotina Diária</div>", unsafe_allow_html=True)

    eid, estudante = selector_estudante("sel_rot")
    if eid:
        tab_lista, tab_add = st.tabs(["Histórico", "Registar Rotina"])

        with tab_lista:
            rotinas = db.listar_rotinas(eid)
            if rotinas:
                df = pd.DataFrame(rotinas)
                df = df.rename(columns={
                    "data":"Data","estudo_h":"Estudo(h)","aulas_h":"Aulas(h)",
                    "descanso_h":"Descanso(h)","lazer_h":"Lazer(h)",
                    "sono_h":"Sono(h)","exercicio_h":"Exercício(h)",
                })
                st.dataframe(df.drop(columns=["id","estudante_id"], errors="ignore"),
                             use_container_width=True, hide_index=True)

                # Gráfico barras empilhadas
                st.subheader("Distribuição Média do Tempo Diário")
                cats = ["estudo_h","aulas_h","descanso_h","lazer_h","sono_h","exercicio_h"]
                nomes_cat = ["Estudo","Aulas","Descanso","Lazer","Sono","Exercício"]
                medias = [np.mean([r[c] for r in rotinas if r.get(c)]) for c in cats]
                fig, ax = plt.subplots(figsize=(7, 3.5))
                _tema_fig(fig, ax)
                bars = ax.barh(nomes_cat, medias, color=PALETTE[:len(cats)], height=0.5)
                ax.set_xlabel("Horas (média)")
                ax.set_title("Tempo Médio por Actividade", color=TEXT)
                for bar, v in zip(bars, medias):
                    ax.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                            f"{v:.1f}h", va="center", color=TEXT, fontsize=9)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Sem rotinas registadas.")

        with tab_add:
            with st.form("form_add_rotina"):
                data = st.date_input("Data", datetime.today(), key="data_rot")
                c1, c2 = st.columns(2)
                with c1:
                    est  = st.number_input("Estudo (h)", 0.0, 12.0, 3.0, step=0.5)
                    aul  = st.number_input("Aulas (h)", 0.0, 10.0, 4.0, step=0.5)
                    desc = st.number_input("Descanso (h)", 0.0, 10.0, 1.5, step=0.5)
                with c2:
                    laz  = st.number_input("Lazer (h)", 0.0, 10.0, 2.0, step=0.5)
                    sono = st.number_input("Sono (h)", 0.0, 12.0, 7.0, step=0.5)
                    exe  = st.number_input("Exercício (h)", 0.0, 5.0, 0.5, step=0.5)
                if st.form_submit_button("💾 Guardar Rotina", type="primary"):
                    db.adicionar_rotina(eid, data.strftime("%Y-%m-%d"),
                                        est, aul, desc, laz, sono, exe)
                    st.success("✅ Rotina registada!")
                    st.rerun()


# ════════════════════════════════════════════════════════════════
# PÁGINA: GRÁFICOS
# ════════════════════════════════════════════════════════════════

elif pagina == "📊  Gráficos":
    st.markdown("<div class='section-title'>📊 Gráficos</div>", unsafe_allow_html=True)

    eid, estudante = selector_estudante("sel_graf")
    if eid:
        sessoes = db.listar_sessoes(eid)
        habitos = db.listar_habitos(eid)
        rotinas = db.listar_rotinas(eid)

        tipo = st.selectbox("Tipo de gráfico", [
            "Evolução das Notas", "Horas de Estudo por Matéria",
            "Concentração ao Longo do Tempo", "Hábitos Digitais (Barras)",
            "Distribuição Distrações (Pizza)", "Radar de Hábitos",
            "Comparativo Notas vs Concentração",
        ])

        fig, ax = plt.subplots(figsize=(10, 4.5))
        _tema_fig(fig, ax)

        if tipo == "Evolução das Notas" and sessoes:
            dados = list(reversed(sessoes[-30:]))
            datas = [s["data"][-5:] for s in dados]
            notas = [s["nota"] or 0 for s in dados]
            ax.plot(datas, notas, color=AZUL, linewidth=2, marker="o", markersize=5, markerfacecolor=CIANO)
            ax.fill_between(range(len(notas)), notas, alpha=0.12, color=AZUL)
            ax.axhline(y=10, color=VERM, linestyle="--", linewidth=1, alpha=0.7, label="Mínimo (10)")
            ax.set_xticks(range(len(datas))); ax.set_xticklabels(datas, rotation=45, fontsize=7)
            ax.set_ylim(0, 20); ax.set_ylabel("Nota")
            ax.set_title("Evolução das Notas", color=TEXT)
            ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=MUTED, framealpha=0.8)

        elif tipo == "Horas de Estudo por Matéria" and sessoes:
            por_mat = {}
            for s in sessoes:
                m = s["materia"] or "Outra"
                por_mat[m] = por_mat.get(m, 0) + (s["duracao_min"] or 0) / 60
            ax.bar(list(por_mat.keys()), list(por_mat.values()), color=PALETTE[:len(por_mat)], width=0.5)
            ax.set_ylabel("Horas"); ax.set_title("Horas Totais por Matéria", color=TEXT)
            ax.tick_params(axis="x", rotation=30)

        elif tipo == "Concentração ao Longo do Tempo" and sessoes:
            dados = list(reversed(sessoes[-30:]))
            datas = [s["data"][-5:] for s in dados]
            concs = [s["concentracao"] or 0 for s in dados]
            ax.plot(datas, concs, color=VERDE, linewidth=2, marker="s", markersize=5)
            ax.fill_between(range(len(concs)), concs, alpha=0.12, color=VERDE)
            ax.set_xticks(range(len(datas))); ax.set_xticklabels(datas, rotation=45, fontsize=7)
            ax.set_ylim(0, 10); ax.set_ylabel("Concentração (1–10)")
            ax.set_title("Nível de Concentração", color=TEXT)

        elif tipo == "Hábitos Digitais (Barras)" and habitos:
            cats = ["whatsapp_h","youtube_h","instagram_h","jogos_h","estudo_online_h"]
            noms = ["WhatsApp","YouTube","Instagram","Jogos","Est.Online"]
            medias = [np.mean([h[c] for h in habitos]) for c in cats]
            ax.bar(noms, medias, color=PALETTE[:5], width=0.5)
            ax.set_ylabel("Horas/dia (média)"); ax.set_title("Hábitos Digitais Médios", color=TEXT)

        elif tipo == "Distribuição Distrações (Pizza)" and habitos:
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(6, 5))
            _tema_fig(fig, ax)
            wh = np.mean([h["whatsapp_h"] for h in habitos])
            yh = np.mean([h["youtube_h"] for h in habitos])
            ig = np.mean([h["instagram_h"] for h in habitos])
            jg = np.mean([h["jogos_h"] for h in habitos])
            ax.pie([wh,yh,ig,jg], labels=["WhatsApp","YouTube","Instagram","Jogos"],
                   autopct="%1.0f%%", colors=PALETTE[:4], startangle=90,
                   textprops={"color":TEXT,"fontsize":9})
            for at in ax.texts: at.set_color(TEXT)
            ax.set_title("Distrações Digitais", color=TEXT)

        elif tipo == "Radar de Hábitos":
            plt.close(fig)
            fig = plt.figure(figsize=(6, 5))
            fig.patch.set_facecolor(DARK_BG)
            ax_r = fig.add_subplot(111, polar=True)
            ax_r.set_facecolor(DARKER)
            ax_r.tick_params(colors=MUTED, labelsize=8)
            cats  = ["Nota","Concentração","Estudo","Sono","Distracção Inv.","Exercício"]
            if sessoes and habitos and rotinas:
                f = predictor.extrair_features(sessoes, habitos, rotinas)
                vals = [
                    f["media_nota"] / 20,
                    f["media_concentracao"] / 10,
                    min(1, f.get("horas_estudo_sem", 0) / 8),
                    min(1, f.get("media_sono", 0) / 9),
                    max(0, 1 - f.get("total_distract", 0) / 12),
                    min(1, f.get("media_exercicio", 0) / 1.5),
                ]
            else:
                vals = [0.5] * 6
            N = len(cats)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            vals_c = vals + [vals[0]]
            angles_c = angles + [angles[0]]
            ax_r.plot(angles_c, vals_c, color=AZUL, linewidth=2)
            ax_r.fill(angles_c, vals_c, color=AZUL, alpha=0.25)
            ax_r.set_xticks(angles); ax_r.set_xticklabels(cats, color=TEXT, fontsize=9)
            ax_r.set_ylim(0, 1); ax_r.set_title("Radar de Hábitos", color=TEXT, pad=15)
            ax_r.grid(color=GRID, alpha=0.6)

        elif tipo == "Comparativo Notas vs Concentração" and sessoes:
            notas = [s["nota"] or 0 for s in sessoes]
            concs = [s["concentracao"] or 0 for s in sessoes]
            ax.scatter(concs, notas, color=ROXO, alpha=0.7, s=50, edgecolors=CIANO, linewidths=0.5)
            ax.set_xlabel("Concentração (1–10)"); ax.set_ylabel("Nota (0–20)")
            ax.set_title("Notas vs Concentração", color=TEXT)
            if len(notas) > 2:
                m, b = np.polyfit(concs, notas, 1)
                xs = np.linspace(min(concs), max(concs), 100)
                ax.plot(xs, m*xs+b, color=AMBER, linestyle="--", linewidth=1.5, alpha=0.7, label=f"y={m:.2f}x+{b:.1f}")
                ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=MUTED, framealpha=0.8)

        else:
            ax.text(0.5, 0.5, "Sem dados suficientes para este gráfico.",
                    ha="center", va="center", color=MUTED, fontsize=12, transform=ax.transAxes)

        st.pyplot(fig)
        plt.close(fig)


# ════════════════════════════════════════════════════════════════
# PÁGINA: PREVISÃO IA
# ════════════════════════════════════════════════════════════════

elif pagina == "🔮  Previsão IA":
    st.markdown("<div class='section-title'>🔮 Previsão IA</div>", unsafe_allow_html=True)

    eid, estudante = selector_estudante("sel_prev")
    if eid:
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            if estudante:
                st.markdown(f"**Analisando:** {estudante['nome']} · {estudante.get('curso','—')} · Ano {estudante.get('ano','—')}")
        with col_btn:
            analisar = st.button("🔮 Analisar Agora", type="primary", use_container_width=True)

        if analisar:
            with st.spinner("A processar análise comportamental…"):
                sessoes = db.listar_sessoes(eid)
                habitos = db.listar_habitos(eid)
                rotinas = db.listar_rotinas(eid)
                res = predictor.analisar_estudante(sessoes, habitos, rotinas)

            f = res["features"]
            perfil_info = res["perfil_info"]
            ml_info     = res["ml_info"]

            # ─── Métricas principais ───
            c1, c2, c3 = st.columns(3)
            rr = res["risco_reprovacao"]
            rb = res["risco_burnout"]
            sp = res["score_produtividade"]
            with c1:
                st.metric("⚠️ Risco de Reprovação", f"{rr*100:.0f}%",
                          delta=f"{'Alto' if rr>0.6 else 'Médio' if rr>0.3 else 'Baixo'}")
            with c2:
                st.metric("🔥 Risco de Burnout", f"{rb*100:.0f}%",
                          delta=f"{'Alto' if rb>0.6 else 'Médio' if rb>0.3 else 'Baixo'}")
            with c3:
                st.metric("⚡ Score de Produtividade", f"{sp:.0f} / 100")

            # ─── Perfil ───
            cor_perfil = perfil_info.get("cor", AZUL)
            st.markdown(f"""
            <div style="background:{cor_perfil}22; border-left: 4px solid {cor_perfil};
                        padding: 14px 18px; border-radius: 8px; margin: 16px 0;">
                <b style="font-size:1.1rem;">Perfil: {res['perfil']}</b><br>
                <span style="color:#94a3b8;">{perfil_info.get('desc','')}</span>
            </div>
            """, unsafe_allow_html=True)

            # ─── Info ML ───
            via = ml_info.get("via", "—")
            acc = ml_info.get("acuracia_modelo")
            st.caption(f"Modelo: **{via}** · Acurácia: {f'{acc*100:.1f}%' if acc else '—'}")

            # ─── Classe ML ───
            classe = ml_info.get("classe", "—")
            probs  = ml_info.get("probs", {})
            st.markdown("##### Previsão do Modelo")
            c1, c2, c3 = st.columns(3)
            for col, (cls, prob) in zip([c1,c2,c3], {
                "aprovado":  probs.get("aprovado", 0),
                "reprovado": probs.get("reprovado", 0),
                "desistiu":  probs.get("desistiu", 0),
            }.items()):
                emoji = "✅" if cls=="aprovado" else ("❌" if cls=="reprovado" else "🚪")
                col.metric(f"{emoji} {cls.capitalize()}", f"{prob*100:.1f}%")

            # ─── Gauge bar visual ───
            fig, axes = plt.subplots(1, 3, figsize=(10, 1.4))
            _tema_fig(fig, axes)
            for ax, val, label, cor_low, cor_hi in [
                (axes[0], rr,     "Risco Reprovação", VERDE, VERM),
                (axes[1], rb,     "Risco Burnout",    VERDE, VERM),
                (axes[2], sp/100, "Produtividade",    VERM,  VERDE),
            ]:
                cor = cor_low if val < 0.4 else (AMBER if val < 0.7 else cor_hi)
                ax.barh([0], [1], color=GRID, height=0.4)
                ax.barh([0], [val], color=cor, height=0.4)
                ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5)
                ax.set_yticks([]); ax.set_xticks([0, 0.5, 1])
                ax.set_xticklabels(["0%","50%","100%"], fontsize=7)
                ax.set_title(label, fontsize=9, color=TEXT, pad=4)
                ax.text(val, 0, f" {val*100:.0f}%", va="center", color=TEXT, fontsize=8, fontweight="bold")
            plt.tight_layout(pad=1.5)
            st.pyplot(fig)
            plt.close(fig)

            # ─── Recomendações ───
            st.markdown("##### 💡 Recomendações Personalizadas")
            for rec in res["recomendacoes"]:
                st.markdown(f"- {rec}")

            # ─── Features breakdown ───
            with st.expander("🔍 Ver todas as features extraídas"):
                feat_df = pd.DataFrame([{
                    "Feature": k.replace("_", " ").title(),
                    "Valor": round(v, 3) if isinstance(v, float) else v,
                } for k, v in f.items()])
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

            # ─── Guardar previsão ───
            db.salvar_previsao(
                eid, via, rr, rb, sp,
                " | ".join(res["recomendacoes"][:2])
            )

        # Histórico de previsões
        st.markdown("---")
        st.subheader("📜 Histórico de Previsões")
        prev_hist = db.listar_previsoes(eid)
        if prev_hist:
            df = pd.DataFrame(prev_hist).rename(columns={
                "data":"Data","modelo":"Modelo",
                "risco_reprovacao":"R.Reprov.","risco_burnout":"R.Burnout",
                "score_produt":"Produtiv.","recomendacao":"Recomendação",
            })
            df["R.Reprov."] = df["R.Reprov."].apply(lambda x: f"{x*100:.0f}%")
            df["R.Burnout"] = df["R.Burnout"].apply(lambda x: f"{x*100:.0f}%")
            df["Produtiv."] = df["Produtiv."].apply(lambda x: f"{x:.0f}/100")
            st.dataframe(df.drop(columns=["id","estudante_id"], errors="ignore"),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Nenhuma previsão guardada ainda. Carrega em **Analisar Agora**.")


# ════════════════════════════════════════════════════════════════
# PÁGINA: PAINEL ML
# ════════════════════════════════════════════════════════════════

elif pagina == "⚙️  Painel ML":
    st.markdown("<div class='section-title'>⚙️ Painel de Gestão ML</div>", unsafe_allow_html=True)

    info = predictor.obter_info_modelo()
    n_reais = db.contar_com_resultado()
    ml_disp = predictor.ML_DISPONIVEL

    # ─── Estado do modelo ───
    st.subheader("Estado do Modelo Activo")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        estado = "✅ Pronto" if info["pronto"] else "⏳ A treinar…"
        st.metric("Estado", estado)
    with c2:
        st.metric("Tipo", info.get("tipo","—").capitalize())
    with c3:
        acc = info.get("acuracia", 0)
        st.metric("Acurácia", f"{acc*100:.1f}%" if acc else "—")
    with c4:
        f1 = info.get("f1", 0)
        st.metric("F1-Score", f"{f1*100:.1f}%" if f1 else "—")

    if info.get("data_treino"):
        st.caption(f"Último treino: {info['data_treino']} · {info.get('n_amostras','?')} amostras")

    if not ml_disp:
        st.warning("⚠️ **scikit-learn não instalado** — o sistema usa modo heurístico. Instala `scikit-learn` para activar o ML completo.")

    st.markdown("---")

    # ─── Fase A ───
    st.subheader("Fase A — Dataset Sintético")
    st.markdown("""
    O modelo é inicializado com **2000 estudantes sintéticos** gerados com base em
    correlações da literatura de aprendizagem automática educacional angolana.
    Usa um **Random Forest** com 200 árvores, StandardScaler e class_weight='balanced'.
    """)
    if st.button("🔄 Re-treinar com Dataset Sintético", disabled=not ml_disp):
        with st.spinner("A treinar… pode demorar alguns segundos."):
            import threading
            predictor._background_treino_sintetico()
        st.success("✅ Modelo sintético re-treinado com sucesso!")
        st.rerun()

    st.markdown("---")

    # ─── Fase B ───
    st.subheader(f"Fase B — Re-treino com Dados Reais ({n_reais} registos)")
    st.markdown(f"""
    Quando há **≥ {predictor.MIN_DADOS_REAIS} estudantes com resultado real** preenchido,
    o modelo pode ser re-treinado mesclando dados reais + sintéticos (âncora).
    """)

    st.progress(min(1.0, n_reais / predictor.MIN_DADOS_REAIS),
                text=f"{n_reais}/{predictor.MIN_DADOS_REAIS} registos reais necessários")

    # Definir resultado real
    with st.expander("📝 Definir Resultado Real de Estudante"):
        eid_b, est_b = selector_estudante("sel_mlb")
        if eid_b:
            resultado = st.selectbox("Resultado final", ["aprovado","reprovado","desistiu"])
            if st.button("💾 Guardar Resultado Real"):
                db.actualizar_resultado_real(eid_b, resultado)
                st.success(f"✅ Resultado '{resultado}' guardado para {est_b['nome']}.")
                st.rerun()

    if n_reais >= predictor.MIN_DADOS_REAIS:
        if st.button("🚀 Re-treinar com Dados Reais (Fase B)", type="primary", disabled=not ml_disp):
            with st.spinner("A treinar modelo misto… pode demorar."):
                registos = db.listar_estudantes_com_resultado()
                resultado = predictor.retreinar_com_dados_reais(registos)
                if "erro" not in resultado:
                    db.registar_treino(
                        resultado.get("tipo","misto"),
                        resultado.get("n_amostras",0),
                        resultado.get("acuracia",0),
                        resultado.get("f1",0),
                    )
                    st.success("✅ Re-treino com dados reais concluído!")
                else:
                    st.error(f"Erro: {resultado['erro']}")
            st.rerun()
    else:
        st.info(f"Precisas de {predictor.MIN_DADOS_REAIS - n_reais} resultados reais adicionais para activar o re-treino.")

    st.markdown("---")

    # ─── Log de treinos ───
    st.subheader("📋 Log de Treinos")
    log = db.listar_treino_log()
    if log:
        df = pd.DataFrame(log).rename(columns={
            "id":"ID","data_treino":"Data","tipo":"Tipo",
            "n_amostras":"Amostras","acuracia":"Acurácia",
            "f1_score":"F1","notas":"Notas",
        })
        df["Acurácia"] = df["Acurácia"].apply(lambda x: f"{x*100:.1f}%" if x else "—")
        df["F1"] = df["F1"].apply(lambda x: f"{x*100:.1f}%" if x else "—")
        st.dataframe(df.drop(columns=["ID"], errors="ignore"),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum treino registado ainda.")
