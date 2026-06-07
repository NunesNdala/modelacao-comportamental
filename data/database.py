"""
Camada de base de dados — SQLite
Gestão de estudantes, sessões, hábitos digitais e previsões
"""

import sqlite3
import os
import random
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "comportamento.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Cria todas as tabelas se não existirem."""
    conn = get_connection()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS estudantes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            nome        TEXT NOT NULL,
            idade       INTEGER,
            curso       TEXT,
            ano         INTEGER,
            criado_em   TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sessoes_estudo (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            estudante_id    INTEGER REFERENCES estudantes(id) ON DELETE CASCADE,
            data            TEXT NOT NULL,
            duracao_min     INTEGER,
            materia         TEXT,
            concentracao    INTEGER,   -- 1-10
            nota            REAL,
            observacoes     TEXT
        );

        CREATE TABLE IF NOT EXISTS habitos_digitais (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            estudante_id    INTEGER REFERENCES estudantes(id) ON DELETE CASCADE,
            data            TEXT NOT NULL,
            whatsapp_h      REAL,
            youtube_h       REAL,
            instagram_h     REAL,
            jogos_h         REAL,
            estudo_online_h REAL,
            periodo_uso     TEXT,      -- manha / tarde / noite
            sono_h          REAL
        );

        CREATE TABLE IF NOT EXISTS rotinas (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            estudante_id    INTEGER REFERENCES estudantes(id) ON DELETE CASCADE,
            data            TEXT NOT NULL,
            estudo_h        REAL,
            aulas_h         REAL,
            descanso_h      REAL,
            lazer_h         REAL,
            sono_h          REAL,
            exercicio_h     REAL
        );

        CREATE TABLE IF NOT EXISTS previsoes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            estudante_id    INTEGER REFERENCES estudantes(id) ON DELETE CASCADE,
            data            TEXT DEFAULT (datetime('now')),
            modelo          TEXT,
            risco_reprovacao REAL,
            risco_burnout   REAL,
            score_produt    REAL,
            recomendacao    TEXT
        );
    """)
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# ESTUDANTES
# ─────────────────────────────────────────────

def listar_estudantes():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM estudantes ORDER BY nome").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def adicionar_estudante(nome, idade, curso, ano):
    conn = get_connection()
    conn.execute(
        "INSERT INTO estudantes (nome, idade, curso, ano) VALUES (?,?,?,?)",
        (nome, idade, curso, ano)
    )
    conn.commit()
    eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return eid


def remover_estudante(eid):
    conn = get_connection()
    conn.execute("DELETE FROM estudantes WHERE id=?", (eid,))
    conn.commit()
    conn.close()


def obter_estudante(eid):
    conn = get_connection()
    row = conn.execute("SELECT * FROM estudantes WHERE id=?", (eid,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ─────────────────────────────────────────────
# SESSÕES DE ESTUDO
# ─────────────────────────────────────────────

def adicionar_sessao(eid, data, duracao, materia, concentracao, nota, obs=""):
    conn = get_connection()
    conn.execute(
        """INSERT INTO sessoes_estudo
           (estudante_id,data,duracao_min,materia,concentracao,nota,observacoes)
           VALUES (?,?,?,?,?,?,?)""",
        (eid, data, duracao, materia, concentracao, nota, obs)
    )
    conn.commit()
    conn.close()


def listar_sessoes(eid):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sessoes_estudo WHERE estudante_id=? ORDER BY data DESC",
        (eid,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# HÁBITOS DIGITAIS
# ─────────────────────────────────────────────

def adicionar_habito(eid, data, wh, yh, ig, jg, eo, periodo, sono):
    conn = get_connection()
    conn.execute(
        """INSERT INTO habitos_digitais
           (estudante_id,data,whatsapp_h,youtube_h,instagram_h,
            jogos_h,estudo_online_h,periodo_uso,sono_h)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (eid, data, wh, yh, ig, jg, eo, periodo, sono)
    )
    conn.commit()
    conn.close()


def listar_habitos(eid):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM habitos_digitais WHERE estudante_id=? ORDER BY data DESC",
        (eid,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# ROTINAS
# ─────────────────────────────────────────────

def adicionar_rotina(eid, data, est, aul, desc, laz, sono, exe):
    conn = get_connection()
    conn.execute(
        """INSERT INTO rotinas
           (estudante_id,data,estudo_h,aulas_h,descanso_h,lazer_h,sono_h,exercicio_h)
           VALUES (?,?,?,?,?,?,?,?)""",
        (eid, data, est, aul, desc, laz, sono, exe)
    )
    conn.commit()
    conn.close()


def listar_rotinas(eid):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM rotinas WHERE estudante_id=? ORDER BY data DESC",
        (eid,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# PREVISÕES
# ─────────────────────────────────────────────

def salvar_previsao(eid, modelo, risco_rep, risco_burn, score_prod, recom):
    conn = get_connection()
    conn.execute(
        """INSERT INTO previsoes
           (estudante_id,modelo,risco_reprovacao,risco_burnout,score_produt,recomendacao)
           VALUES (?,?,?,?,?,?)""",
        (eid, modelo, risco_rep, risco_burn, score_prod, recom)
    )
    conn.commit()
    conn.close()


def listar_previsoes(eid):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM previsoes WHERE estudante_id=? ORDER BY data DESC LIMIT 10",
        (eid,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# DADOS DEMO
# ─────────────────────────────────────────────

def popular_dados_demo():
    """Insere dados de exemplo para demonstração."""
    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) FROM estudantes").fetchone()[0]
    conn.close()
    if n > 0:
        return  # já tem dados

    estudantes = [
        ("Angelina Santos", 20, "Eng. Informática", 2),
        ("João Tchiweyengue", 21, "Eng. Informática", 2),
        ("Josimar Carlos", 22, "Eng. Informática", 3),
        ("Maria Fernanda", 19, "Eng. Informática", 1),
        ("Pedro Muangala", 23, "Eng. Informática", 3),
    ]

    materias = ["Algoritmos", "BD", "POO", "Redes", "Matemática", "SO"]
    periodos = ["manha", "tarde", "noite"]

    for nome, idade, curso, ano in estudantes:
        eid = adicionar_estudante(nome, idade, curso, ano)
        base_nota = random.uniform(8, 16)
        base_conc = random.randint(5, 9)

        for i in range(30):
            d = (datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d")
            dur = random.randint(30, 180)
            conc = max(1, min(10, base_conc + random.randint(-2, 2)))
            nota = max(0, min(20, base_nota + random.uniform(-3, 3)))
            adicionar_sessao(eid, d, dur, random.choice(materias), conc, round(nota, 1))

        for i in range(30):
            d = (datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d")
            adicionar_habito(
                eid, d,
                round(random.uniform(0.5, 3), 1),
                round(random.uniform(0.5, 4), 1),
                round(random.uniform(0.3, 2.5), 1),
                round(random.uniform(0, 2), 1),
                round(random.uniform(0.5, 2), 1),
                random.choice(periodos),
                round(random.uniform(5, 9), 1)
            )

        for i in range(30):
            d = (datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d")
            adicionar_rotina(
                eid, d,
                round(random.uniform(1, 5), 1),
                round(random.uniform(3, 6), 1),
                round(random.uniform(1, 3), 1),
                round(random.uniform(0.5, 3), 1),
                round(random.uniform(5, 9), 1),
                round(random.uniform(0, 1.5), 1)
            )


# ═══════════════════════════════════════════════════════════════
# FASE B — Resultado real e re-treino
# ═══════════════════════════════════════════════════════════════

def adicionar_campo_resultado_real():
    """Migração: adiciona colunas resultado_real e data_resultado se não existirem."""
    conn = get_connection()
    try:
        conn.execute("ALTER TABLE estudantes ADD COLUMN resultado_real TEXT")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE estudantes ADD COLUMN data_resultado TEXT")
    except Exception:
        pass
    # Tabela de log de treino
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS treino_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            data_treino TEXT DEFAULT (datetime('now')),
            tipo        TEXT,
            n_amostras  INTEGER,
            acuracia    REAL,
            f1_score    REAL,
            notas       TEXT
        );
    """)
    conn.commit()
    conn.close()


def actualizar_resultado_real(eid: int, resultado: str):
    """Define o resultado real do estudante para alimentar o re-treino."""
    from datetime import datetime as _dt
    conn = get_connection()
    conn.execute(
        "UPDATE estudantes SET resultado_real=?, data_resultado=? WHERE id=?",
        (resultado, _dt.now().strftime("%Y-%m-%d %H:%M:%S"), eid)
    )
    conn.commit()
    conn.close()


def listar_estudantes_com_resultado():
    """Devolve todos os estudantes com resultado_real preenchido (Fase B)."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM estudantes WHERE resultado_real IS NOT NULL ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def contar_com_resultado() -> int:
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) as n FROM estudantes WHERE resultado_real IS NOT NULL"
    ).fetchone()
    conn.close()
    return row["n"]


def registar_treino(tipo: str, n_amostras: int, acuracia: float,
                    f1: float, notas: str = ""):
    conn = get_connection()
    conn.execute(
        """INSERT INTO treino_log (tipo, n_amostras, acuracia, f1_score, notas)
           VALUES (?,?,?,?,?)""",
        (tipo, n_amostras, round(acuracia, 4), round(f1, 4), notas)
    )
    conn.commit()
    conn.close()


def listar_treino_log():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM treino_log ORDER BY id DESC LIMIT 30"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
