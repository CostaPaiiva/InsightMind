# 🧠 InsightMind

Dashboard automático para análise de datasets (CSV) com **qualidade de dados**, **diagnóstico**, **gráficos**, **limpeza** e **relatórios (HTML/PDF)** — tudo em **Streamlit**.

> Ideal para: explorar rapidamente um CSV, identificar problemas (missing, duplicadas, colunas constantes), visualizar distribuições/correlação e gerar relatórios para compartilhar.

---

## ✅ Funcionalidades

### 1) Upload inteligente de CSV
- Carregamento com detecção de separador/encoding (via `load_csv_smart`).
- Preview configurável (slider no sidebar).

### 2) Resumo + Qualidade
- **Resumo por coluna**: tipo, % missing, n_unique, exemplo.
- **Métricas de qualidade**:
  - missing total e %
  - linhas duplicadas
  - colunas numéricas/categóricas
  - colunas constantes

### 3) Gráficos (Plotly)
- Histogramas (numéricas)
- Correlação (numéricas) com limite de colunas para evitar travar
- Barras (categóricas) com proteção contra cardinalidade muito alta  
➡️ Os gráficos são gerados **somente ao clicar no botão** para evitar lentidão.

### 4) Diagnóstico Automático
- Mostra:
  - métricas de qualidade (JSON)
  - resumo estatístico (top 30)
  - principais problemas (prioridade)
  - insights automáticos (heurísticos)
  - recomendações práticas

### 5) Limpeza de dados (pipeline)
- Remover duplicadas
- Padronizar strings
- Converter datas
- Remover colunas com missing alto
- Imputação numérica/categórica
- Remover colunas constantes
- Clip de outliers (IQR)
- Download do CSV tratado

### 6) Relatórios HTML e PDF
- HTML interativo
- PDF com imagens de gráficos principais (export via Plotly)
- Geração de figuras e PDF **somente no clique**, para performance

---

## 🧱 Estrutura do Projeto

InsightMind/
├─ app.py
├─ core/
│ ├─ loader.py
│ ├─ profiler.py
│ ├─ insights.py
│ ├─ visuals.py
│ ├─ cleaning.py
│ ├─ report.py
├─ requirements.txt
└─ README.md


---

## 🚀 Como rodar localmente

### 1) Criar ambiente virtual
**Windows**
```bash
python -m venv venv
venv\Scripts\activate
Linux/Mac

python -m venv venv
source venv/bin/activate
2) Instalar dependências
pip install -r requirements.txt
3) Rodar o app
streamlit run app.py
Acesse:

http://localhost:8501


🧠 Como usar
Abra o app

Faça upload de um .csv

Veja o preview e as abas:

Resumo: estatísticas e métricas

Gráficos: clique em “Gerar gráficos”

Diagnóstico: leitura automática e recomendações

Limpeza: configure e aplique um pipeline

Relatório: gere HTML/PDF e baixe

⚡ Performance (importante)
O app foi pensado para não travar com datasets grandes:

Leitura do CSV cacheada com @st.cache_data

Métricas/resumo/insights cacheados

Gráficos só geram quando você clicar

Amostragem automática para gráficos (ex.: até 20k linhas)

Correlação limitada (ex.: até 25 colunas numéricas)

Categóricas com cardinalidade muito alta são puladas

Se ainda estiver lento:

Use CSV menor, ou

Limite colunas no load_csv_smart, ou

Aumente amostragem/limites nos arquivos core/visuals.py e core/profiler.py

🧪 Possíveis melhorias (roadmap)
Exportar relatório como .docx

Detecção de outliers e sugestões automáticas por coluna

Perfil completo (ydata-profiling) opcional em aba separada

Cache por hash do arquivo (mais robusto)

Deploy no Streamlit Cloud

🛠️ Troubleshooting
Erro no PDF / imagens Plotly
Se aparecer erro no to_image():

pip install -U kaleido
App lento
Garanta que os gráficos só rodam no clique

Reduza preview

Use amostragem no visuals.py

CSV com encoding estranho
O load_csv_smart tenta detectar o encoding. Se ainda falhar, salve o CSV como UTF-8.

