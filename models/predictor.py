"""
Motor de Modelação Comportamental — ML Pipeline Completo
========================================================
Fase A: Dataset sintético (2000 estudantes) + Random Forest treinado no arranque
Fase B: Re-treino automático quando há dados reais suficientes (≥ 50 registos)

O modelo começa com dados sintéticos e substitui gradualmente por dados reais.
"""

import numpy as np
import os
import threading
from datetime import datetime

# ─── imports ML (opcionais — degrada graciosamente se ausentes) ───
try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.pipeline import Pipeline
    ML_DISPONIVEL = True
except ImportError:
    ML_DISPONIVEL = False

# ─── Caminho do modelo persistido ───
_DIR     = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(_DIR, "rf_model.joblib")
META_PATH   = os.path.join(_DIR, "model_meta.joblib")

# ─── Estado global do modelo ───
_modelo_info = {
    "modelo":      None,
    "scaler":      None,
    "tipo":        "nenhum",   # sintetico | real | misto
    "n_amostras":  0,
    "acuracia":    0.0,
    "f1":          0.0,
    "data_treino": None,
    "pronto":      False,
}
_lock = threading.Lock()

# ── Limiar para activar Fase B ──
MIN_DADOS_REAIS = 50


# ═══════════════════════════════════════════════════════════════
# GERAÇÃO DO DATASET SINTÉTICO (Fase A)
# ═══════════════════════════════════════════════════════════════

def _gerar_dataset_sintetico(n: int = 2000, seed: int = 42):
    """
    Gera um dataset realista de estudantes angolanos com base em
    correlações da literatura de aprendizagem automática educacional.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    for _ in range(n):
        # Perfil base
        perfil = rng.choice(["excelente","bom","médio","fraco","risco"],
                             p=[0.15, 0.25, 0.30, 0.20, 0.10])

        if perfil == "excelente":
            media_nota       = rng.uniform(15, 20)
            frequencia       = rng.uniform(0.85, 1.0)
            horas_estudo     = rng.uniform(5, 8)
            concentracao     = rng.uniform(7.5, 10)
            sono             = rng.uniform(7, 9)
            distrac          = rng.uniform(1, 3)
            lazer            = rng.uniform(1.5, 3)
            exercicio        = rng.uniform(0.5, 1.5)
            apoio_financeiro = rng.integers(0, 2)
            trabalha         = 0
            saude_mental     = rng.integers(7, 11)
            participacao     = rng.integers(7, 11)
            variancia_notas  = rng.uniform(0.5, 3)
            resultado        = "aprovado"

        elif perfil == "bom":
            media_nota       = rng.uniform(12, 16)
            frequencia       = rng.uniform(0.75, 0.90)
            horas_estudo     = rng.uniform(3, 6)
            concentracao     = rng.uniform(5.5, 8)
            sono             = rng.uniform(6.5, 8.5)
            distrac          = rng.uniform(2, 5)
            lazer            = rng.uniform(1, 3)
            exercicio        = rng.uniform(0.3, 1.0)
            apoio_financeiro = rng.integers(0, 2)
            trabalha         = rng.integers(0, 2)
            saude_mental     = rng.integers(6, 10)
            participacao     = rng.integers(5, 9)
            variancia_notas  = rng.uniform(1, 5)
            resultado        = "aprovado"

        elif perfil == "médio":
            media_nota       = rng.uniform(9, 13)
            frequencia       = rng.uniform(0.60, 0.80)
            horas_estudo     = rng.uniform(2, 4)
            concentracao     = rng.uniform(4, 7)
            sono             = rng.uniform(5.5, 8)
            distrac          = rng.uniform(3, 7)
            lazer            = rng.uniform(1, 4)
            exercicio        = rng.uniform(0, 0.8)
            apoio_financeiro = rng.integers(0, 2)
            trabalha         = rng.integers(0, 2)
            saude_mental     = rng.integers(4, 8)
            participacao     = rng.integers(3, 7)
            variancia_notas  = rng.uniform(2, 8)
            resultado        = rng.choice(["aprovado","reprovado"], p=[0.65, 0.35])

        elif perfil == "fraco":
            media_nota       = rng.uniform(5, 10)
            frequencia       = rng.uniform(0.40, 0.65)
            horas_estudo     = rng.uniform(0.5, 2.5)
            concentracao     = rng.uniform(2, 5.5)
            sono             = rng.uniform(4.5, 7)
            distrac          = rng.uniform(5, 10)
            lazer            = rng.uniform(2, 5)
            exercicio        = rng.uniform(0, 0.5)
            apoio_financeiro = rng.integers(0, 2)
            trabalha         = rng.integers(0, 2)
            saude_mental     = rng.integers(2, 6)
            participacao     = rng.integers(1, 5)
            variancia_notas  = rng.uniform(3, 12)
            resultado        = rng.choice(["aprovado","reprovado","desistiu"],
                                           p=[0.30, 0.55, 0.15])
        else:  # risco
            media_nota       = rng.uniform(0, 7)
            frequencia       = rng.uniform(0.20, 0.50)
            horas_estudo     = rng.uniform(0, 1.5)
            concentracao     = rng.uniform(1, 4)
            sono             = rng.uniform(3, 6)
            distrac          = rng.uniform(6, 12)
            lazer            = rng.uniform(3, 6)
            exercicio        = rng.uniform(0, 0.3)
            apoio_financeiro = rng.integers(0, 2)
            trabalha         = rng.integers(0, 2)
            saude_mental     = rng.integers(1, 4)
            participacao     = rng.integers(0, 3)
            variancia_notas  = rng.uniform(5, 18)
            resultado        = rng.choice(["reprovado","desistiu"], p=[0.60, 0.40])

        # Adicionar ruído realista
        media_nota   = np.clip(media_nota + rng.normal(0, 0.8), 0, 20)
        concentracao = np.clip(concentracao + rng.normal(0, 0.5), 1, 10)
        sono         = np.clip(sono + rng.normal(0, 0.4), 0, 12)
        distrac      = np.clip(distrac + rng.normal(0, 0.5), 0, 12)
        horas_estudo = np.clip(horas_estudo + rng.normal(0, 0.3), 0, 12)

        # Feature vector (12 features)
        feats = [
            media_nota,
            frequencia,
            horas_estudo,
            concentracao,
            sono,
            distrac,
            lazer,
            exercicio,
            float(apoio_financeiro),
            float(trabalha),
            float(saude_mental),
            float(participacao),
        ]
        X.append(feats)
        y.append(resultado)

    return np.array(X, dtype=np.float32), np.array(y)


# ═══════════════════════════════════════════════════════════════
# TREINO DO MODELO
# ═══════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "media_nota", "frequencia", "horas_estudo", "media_concentracao",
    "media_sono", "total_distract", "media_lazer", "media_exercicio",
    "apoio_financeiro", "trabalha", "saude_mental", "participacao",
]


def _treinar(X, y, tipo: str) -> dict:
    """Treina o pipeline RF + StandardScaler e devolve métricas."""
    if len(X) < 20:
        return {"erro": "Dados insuficientes para treino"}

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    reporte = classification_report(y_te, y_pred, zero_division=0)

    # Persistir
    joblib.dump(pipeline, MODEL_PATH)
    meta = {
        "tipo": tipo, "n_amostras": len(X),
        "acuracia": acc, "f1": f1,
        "data_treino": datetime.now().isoformat(timespec="seconds"),
        "reporte": reporte,
    }
    joblib.dump(meta, META_PATH)

    return {"pipeline": pipeline, **meta}


def _background_treino_sintetico():
    """Treina o modelo sintético numa thread de background."""
    global _modelo_info
    if not ML_DISPONIVEL:
        return
    try:
        X, y = _gerar_dataset_sintetico(2000)
        resultado = _treinar(X, y, "sintetico")
        if "erro" not in resultado:
            with _lock:
                _modelo_info["modelo"]      = resultado["pipeline"]
                _modelo_info["tipo"]        = "sintetico"
                _modelo_info["n_amostras"]  = resultado["n_amostras"]
                _modelo_info["acuracia"]    = resultado["acuracia"]
                _modelo_info["f1"]          = resultado["f1"]
                _modelo_info["data_treino"] = resultado["data_treino"]
                _modelo_info["pronto"]      = True
    except Exception as e:
        print(f"[ML] Erro no treino sintético: {e}")


def inicializar_modelo():
    """
    Ponto de entrada: carrega modelo persistido ou treina do zero.
    Chamado no arranque da aplicação (non-blocking).
    """
    global _modelo_info
    if not ML_DISPONIVEL:
        return

    # Tentar carregar modelo já existente
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        try:
            pipeline = joblib.load(MODEL_PATH)
            meta     = joblib.load(META_PATH)
            with _lock:
                _modelo_info["modelo"]      = pipeline
                _modelo_info["tipo"]        = meta.get("tipo", "desconhecido")
                _modelo_info["n_amostras"]  = meta.get("n_amostras", 0)
                _modelo_info["acuracia"]    = meta.get("acuracia", 0)
                _modelo_info["f1"]          = meta.get("f1", 0)
                _modelo_info["data_treino"] = meta.get("data_treino", "")
                _modelo_info["pronto"]      = True
            return
        except Exception:
            pass  # Modelo corrompido → retreinar

    # Treinar em background (não bloqueia a GUI)
    t = threading.Thread(target=_background_treino_sintetico, daemon=True)
    t.start()


# ═══════════════════════════════════════════════════════════════
# FASE B — RE-TREINO COM DADOS REAIS
# ═══════════════════════════════════════════════════════════════

def _registos_para_XY(registos) -> tuple:
    """Converte registos da BD em arrays numpy (X, y)."""
    X, y = [], []
    for r in registos:
        try:
            # Calcular total_distract a partir dos dados da BD
            # (os campos podem não existir — usa 0 como fallback)
            def _f(row, key, default=0.0):
                v = dict(row).get(key)
                return float(v) if v is not None else default

            feats = [
                _f(r, "media_nota") or _f(r, "nota_entrada", 10),
                _f(r, "frequencia", 0.8),
                _f(r, "horas_estudo", 3),
                _f(r, "media_concentracao", 6),
                _f(r, "media_sono", 7),
                _f(r, "total_distract", 3),
                _f(r, "media_lazer", 2),
                _f(r, "media_exercicio", 0.5),
                _f(r, "apoio_financeiro"),
                _f(r, "trabalha"),
                _f(r, "saude_mental", 7),
                _f(r, "participacao", 6),
            ]
            resultado = dict(r).get("resultado_real", "")
            if resultado in ("aprovado", "reprovado", "desistiu"):
                X.append(feats)
                y.append(resultado)
        except Exception:
            continue
    return np.array(X, dtype=np.float32) if X else np.array([]), np.array(y)


def retreinar_com_dados_reais(registos_reais):
    """
    Fase B: mescla dados sintéticos (âncora) com dados reais e retreina.
    Chamado em background pela GUI quando há ≥ MIN_DADOS_REAIS.
    """
    global _modelo_info
    if not ML_DISPONIVEL:
        return {"erro": "scikit-learn não disponível"}

    X_real, y_real = _registos_para_XY(registos_reais)
    n_real = len(X_real)
    if n_real < MIN_DADOS_REAIS:
        return {"erro": f"Apenas {n_real} registos reais (mínimo: {MIN_DADOS_REAIS})"}

    # Dados sintéticos como âncora (500 amostras)
    X_sint, y_sint = _gerar_dataset_sintetico(500, seed=99)

    # Combinar: reais têm peso dobrado via duplicação
    X_combined = np.vstack([X_sint, X_real, X_real])
    y_combined  = np.concatenate([y_sint, y_real, y_real])

    tipo = "real" if n_real >= 200 else "misto"
    resultado = _treinar(X_combined, y_combined, tipo)

    if "erro" not in resultado:
        with _lock:
            _modelo_info["modelo"]      = resultado["pipeline"]
            _modelo_info["tipo"]        = tipo
            _modelo_info["n_amostras"]  = len(X_combined)
            _modelo_info["acuracia"]    = resultado["acuracia"]
            _modelo_info["f1"]          = resultado["f1"]
            _modelo_info["data_treino"] = resultado["data_treino"]
            _modelo_info["pronto"]      = True

    return resultado


def obter_info_modelo() -> dict:
    """Devolve estado actual do modelo (thread-safe)."""
    with _lock:
        return dict(_modelo_info)


# ═══════════════════════════════════════════════════════════════
# EXTRACÇÃO DE FEATURES (pipeline existente — compatibilidade)
# ═══════════════════════════════════════════════════════════════

def _safe_mean(lst, default=0.0):
    return float(np.mean(lst)) if lst else default


def _normaliza(val, minv, maxv):
    if maxv == minv:
        return 0.5
    return max(0.0, min(1.0, (val - minv) / (maxv - minv)))


def extrair_features(sessoes, habitos, rotinas):
    f = {}

    if sessoes:
        f["media_duracao"]      = _safe_mean([s["duracao_min"] for s in sessoes])
        f["media_concentracao"] = _safe_mean([s["concentracao"] for s in sessoes])
        f["media_nota"]         = _safe_mean([s["nota"] for s in sessoes if s["nota"] is not None])
        f["total_sessoes"]      = len(sessoes)
        f["horas_estudo_sem"]   = f["media_duracao"] * f["total_sessoes"] / 60 / 4
        notas = [s["nota"] for s in sessoes if s["nota"] is not None]
        f["variancia_notas"]    = float(np.var(notas)) if len(notas) > 1 else 0.0
    else:
        f.update({"media_duracao":0,"media_concentracao":0,"media_nota":0,
                  "total_sessoes":0,"horas_estudo_sem":0,"variancia_notas":0})

    if habitos:
        f["media_whatsapp"]     = _safe_mean([h["whatsapp_h"] for h in habitos])
        f["media_youtube"]      = _safe_mean([h["youtube_h"] for h in habitos])
        f["media_instagram"]    = _safe_mean([h["instagram_h"] for h in habitos])
        f["media_jogos"]        = _safe_mean([h["jogos_h"] for h in habitos])
        f["media_estudo_online"]= _safe_mean([h["estudo_online_h"] for h in habitos])
        f["media_sono"]         = _safe_mean([h["sono_h"] for h in habitos])
        f["total_distract"]     = (f["media_whatsapp"] + f["media_youtube"] +
                                   f["media_instagram"] + f["media_jogos"])
        noites = [h for h in habitos if h["periodo_uso"] == "noite"]
        f["pct_uso_noturno"]    = len(noites) / len(habitos)
    else:
        f.update({"media_whatsapp":0,"media_youtube":0,"media_instagram":0,
                  "media_jogos":0,"media_estudo_online":0,"media_sono":7,
                  "total_distract":0,"pct_uso_noturno":0})

    if rotinas:
        f["media_estudo_rot"] = _safe_mean([r["estudo_h"] for r in rotinas])
        f["media_aulas"]      = _safe_mean([r["aulas_h"] for r in rotinas])
        f["media_lazer"]      = _safe_mean([r["lazer_h"] for r in rotinas])
        f["media_sono_rot"]   = _safe_mean([r["sono_h"] for r in rotinas])
        f["media_exercicio"]  = _safe_mean([r["exercicio_h"] for r in rotinas])
    else:
        f.update({"media_estudo_rot":0,"media_aulas":0,"media_lazer":0,
                  "media_sono_rot":7,"media_exercicio":0})

    return f


# ═══════════════════════════════════════════════════════════════
# PREVISÃO COM ML (substituí a heurística)
# ═══════════════════════════════════════════════════════════════

def _features_para_vetor(f: dict) -> np.ndarray:
    """Converte dict de features no vector esperado pelo modelo."""
    sono = f.get("media_sono") or f.get("media_sono_rot", 7)
    lazer = f.get("media_lazer", 2)
    exercicio = f.get("media_exercicio", 0.5)
    return np.array([[
        f.get("media_nota", 10),
        min(1.0, f.get("horas_estudo_sem", 3) / 7),   # normaliza frequência proxy
        f.get("horas_estudo_sem", 3),
        f.get("media_concentracao", 6),
        sono,
        f.get("total_distract", 3),
        lazer,
        exercicio,
        0.0,  # apoio_financeiro (desconhecido aqui)
        0.0,  # trabalha
        7.0,  # saude_mental (default)
        6.0,  # participacao (default)
    ]], dtype=np.float32)


def prever_com_ml(f: dict) -> dict:
    """
    Usa o modelo ML treinado se disponível.
    Devolve: {"classe": str, "prob_reprovado": float, "probs": dict, "via": str}
    """
    info = obter_info_modelo()
    if ML_DISPONIVEL and info["pronto"] and info["modelo"] is not None:
        try:
            pipeline = info["modelo"]
            X = _features_para_vetor(f)
            classe    = pipeline.predict(X)[0]
            probs_arr = pipeline.predict_proba(X)[0]
            classes   = pipeline.classes_

            prob_dict = {c: float(p) for c, p in zip(classes, probs_arr)}
            prob_rep  = prob_dict.get("reprovado", 0) + prob_dict.get("desistiu", 0) * 0.5

            return {
                "classe":          classe,
                "prob_reprovado":  round(prob_rep, 3),
                "probs":           prob_dict,
                "via":             f"RF ({info['tipo']})",
                "acuracia_modelo": info["acuracia"],
            }
        except Exception as e:
            pass  # fallback para heurística

    # Fallback: heurística original
    prob = prever_risco_reprovacao_heuristico(f)
    if prob > 0.6:
        classe = "reprovado"
    elif prob > 0.35:
        classe = "aprovado"  # borderline
    else:
        classe = "aprovado"

    return {
        "classe":          classe,
        "prob_reprovado":  prob,
        "probs":           {"aprovado": 1-prob, "reprovado": prob},
        "via":             "heurístico",
        "acuracia_modelo": None,
    }


# ═══════════════════════════════════════════════════════════════
# MODELOS HEURÍSTICOS (mantidos como fallback)
# ═══════════════════════════════════════════════════════════════

def prever_risco_reprovacao_heuristico(f):
    score, pesos = 0.0, 0.0
    if f["media_nota"] > 0:
        score += _normaliza(20 - f["media_nota"], 0, 20) * 0.35; pesos += 0.35
    if f["media_concentracao"] > 0:
        score += _normaliza(10 - f["media_concentracao"], 0, 10) * 0.20; pesos += 0.20
    score += _normaliza(max(0, 10 - f["horas_estudo_sem"]), 0, 10) * 0.20; pesos += 0.20
    score += _normaliza(f["total_distract"], 0, 12) * 0.15; pesos += 0.15
    score += _normaliza(max(0, 7 - f["media_sono"]), 0, 7) * 0.10; pesos += 0.10
    return round(score / pesos if pesos > 0 else 0.5, 3)


# Mantém nome original para compatibilidade com código existente
def prever_risco_reprovacao(f):
    return prever_risco_reprovacao_heuristico(f)


def prever_risco_burnout(f):
    score, pesos = 0.0, 0.0
    excesso = max(0, f["horas_estudo_sem"] - 6)
    score += _normaliza(excesso, 0, 10) * 0.25; pesos += 0.25
    score += _normaliza(max(0, 2 - f["media_lazer"]), 0, 2) * 0.20; pesos += 0.20
    score += _normaliza(max(0, 7 - f["media_sono"]), 0, 7) * 0.25; pesos += 0.25
    score += _normaliza(max(0, 0.5 - f["media_exercicio"]), 0, 0.5) * 0.15; pesos += 0.15
    score += f["pct_uso_noturno"] * 0.15; pesos += 0.15
    return round(score / pesos if pesos > 0 else 0.5, 3)


def calcular_score_produtividade(f):
    score = 0.0
    score += (f["media_nota"] / 20) * 30
    score += (f["media_concentracao"] / 10) * 25
    est_dia = f["media_estudo_rot"] or (f["horas_estudo_sem"] / 7)
    if 3 <= est_dia <= 5:   score += 20
    elif est_dia < 3:        score += (est_dia / 3) * 20
    else:                    score += max(0, 20 - (est_dia - 5) * 4)
    sono = f["media_sono"] or f["media_sono_rot"]
    if 7 <= sono <= 8.5:    score += 15
    elif sono < 7:           score += (sono / 7) * 15
    else:                    score += max(0, 15 - (sono - 8.5) * 5)
    score += max(0, 10 - f["total_distract"])
    return round(min(100, max(0, score)), 1)


PERFIS = {
    "Estudante Equilibrado":   {"desc": "Bom balanço entre estudo, descanso e vida social.", "cor": "#10b981"},
    "Workaholic Académico":    {"desc": "Estuda muito mas arrisca esgotamento.", "cor": "#f59e0b"},
    "Distraído Digital":       {"desc": "Muitas horas em redes sociais afectam a concentração.", "cor": "#ef4444"},
    "Em Risco":                {"desc": "Indicadores de baixo desempenho e possível reprovação.", "cor": "#dc2626"},
    "Perfil em Desenvolvimento":{"desc": "Dados insuficientes. Continue a registar.", "cor": "#6366f1"},
}


def classificar_perfil(f, risco_rep, risco_burn, score_prod):
    if f["total_sessoes"] < 5:   return "Perfil em Desenvolvimento"
    if risco_rep > 0.6:          return "Em Risco"
    if risco_burn > 0.6:         return "Workaholic Académico"
    if f["total_distract"] > 6:  return "Distraído Digital"
    return "Estudante Equilibrado"


def gerar_recomendacoes(f, risco_rep, risco_burn, score_prod):
    recom = []
    if risco_rep > 0.5:              recom.append("⚠️ RISCO DE REPROVAÇÃO: Aumenta as horas de estudo e revê os conteúdos mais fracos.")
    if f["media_concentracao"] < 6:  recom.append("🧠 Experimenta a Técnica Pomodoro: 25 min de foco + 5 min de pausa.")
    if f["media_sono"] < 7:          recom.append("😴 Dormes menos de 7h. O sono é essencial para a memória.")
    if f["total_distract"] > 5:      recom.append("📵 Reduz o tempo em redes sociais. Usa o modo 'Não Perturbar' durante o estudo.")
    if f["pct_uso_noturno"] > 0.5:   recom.append("🌙 Mais de 50% do uso do telemóvel é nocturno. Prejudica a qualidade do sono.")
    if risco_burn > 0.6:             recom.append("🔥 Sinais de burnout! Inclui descanso e lazer no horário.")
    if f["media_exercicio"] < 0.3:   recom.append("🏃 Faz pelo menos 30 min de exercício por dia. Melhora o foco.")
    if f["media_estudo_online"] > 1.5: recom.append("✅ Bom uso de plataformas educativas online! Mantém esse hábito.")
    if score_prod > 75:              recom.append("🌟 Excelente produtividade! Continua com as tuas rotinas.")
    if not recom:                    recom.append("✅ O teu perfil está equilibrado. Mantém as tuas rotinas saudáveis.")
    return recom


def analisar_estudante(sessoes, habitos, rotinas):
    """Ponto de entrada principal — compatível com código existente."""
    features = extrair_features(sessoes, habitos, rotinas)

    ml_result  = prever_com_ml(features)
    risco_rep  = ml_result["prob_reprovado"]
    risco_burn = prever_risco_burnout(features)
    score_prod = calcular_score_produtividade(features)
    perfil     = classificar_perfil(features, risco_rep, risco_burn, score_prod)
    recom      = gerar_recomendacoes(features, risco_rep, risco_burn, score_prod)

    return {
        "features":            features,
        "risco_reprovacao":    risco_rep,
        "risco_burnout":       risco_burn,
        "score_produtividade": score_prod,
        "perfil":              perfil,
        "perfil_info":         PERFIS.get(perfil, PERFIS["Perfil em Desenvolvimento"]),
        "recomendacoes":       recom,
        "data_analise":        datetime.now().strftime("%d/%m/%Y %H:%M"),
        "ml_info":             ml_result,
    }


# ── Inicializa modelo no import ──
inicializar_modelo()
