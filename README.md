# Sistema de Modelação Comportamental — Streamlit
### Universidade Mandume Ya N'Demufayo · Instituto Politécnico da Huíla
**Curso:** Engenharia Informática · **Ano:** 2026  
**Autores:** Angelina dos Santos · João Tchiweyengue · Josimar Carlos

---

## Deploy no Render (passo a passo)

### 1. Criar repositório GitHub
```bash
git init
git add .
git commit -m "Sistema de Modelação Comportamental — Streamlit v1.0"
git remote add origin https://github.com/SEU_USUARIO/modelacao-comportamental.git
git push -u origin main
```

### 2. Criar Web Service no Render
1. Acede a [render.com](https://render.com) e faz login
2. Clica em **New → Web Service**
3. Liga o teu repositório GitHub
4. Preenche as configurações:
   - **Name:** `modelacao-comportamental`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Clica em **Create Web Service**

> O Render detecta automaticamente o `render.yaml` — podes também usar **New → Blueprint** e apontar para o repositório.

### 3. Notas importantes para o Render

#### Base de dados persistente
O SQLite por defeito guarda em `/data/database.py` (caminho relativo ao ficheiro).  
No Render, o sistema de ficheiros é **efémero** — os dados perdem-se ao reiniciar.  
Para persistência real, tens duas opções:

**Opção A — Render Disk (recomendado para demo)**  
No painel do serviço: **Disks → Add Disk**
- Mount Path: `/data`
- Size: 1 GB (gratuito no plano pago)

Depois altera em `data/database.py`:
```python
DB_PATH = "/data/comportamento.db"
```

**Opção B — PostgreSQL (produção)**  
Substitui o SQLite por `psycopg2` + variável de ambiente `DATABASE_URL`.

#### Modelo ML persistido
Os ficheiros `rf_model.joblib` e `model_meta.joblib` também precisam do disco persistente.  
Altera em `models/predictor.py`:
```python
_DIR       = "/data"
MODEL_PATH = "/data/rf_model.joblib"
META_PATH  = "/data/model_meta.joblib"
```

---

## Estrutura do projecto

```
streamlit_app/
├── app.py                    ← Aplicação Streamlit principal
├── requirements.txt
├── render.yaml               ← Configuração automática do Render
├── .streamlit/
│   └── config.toml           ← Tema dark + porta 10000
├── data/
│   └── database.py           ← SQLite (estudantes, sessões, hábitos, rotinas, previsões)
└── models/
    └── predictor.py          ← Motor de previsão ML (Random Forest)
```

## Executar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre em: http://localhost:8501

## Funcionalidades

| Página | Descrição |
|--------|-----------|
| 🏠 Dashboard | Métricas globais + tabela resumo + gráfico de produtividade |
| 👥 Estudantes | CRUD completo com abas Lista / Adicionar / Remover |
| 📚 Sessões de Estudo | Registo + histórico em tabela |
| 📱 Hábitos Digitais | Registo + gráfico pizza de distrações |
| 🗓️ Rotina Diária | Registo + gráfico barras horizontais |
| 📊 Gráficos | 7 tipos: linhas, barras, pizza, scatter, radar |
| 🔮 Previsão IA | Random Forest + gauges + recomendações + histórico |
| ⚙️ Painel ML | Estado do modelo, Fase A/B, log de treinos |
