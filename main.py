import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CARREGAR DADOS
# =========================

ratings = pd.read_csv("dados/ratings.csv")
movies = pd.read_csv("dados/movies.csv")

# Junta tabelas
dados = pd.merge(ratings, movies, on="movieId")

# =========================
# MATRIZ USUÁRIO-FILME
# =========================

matriz = dados.pivot_table(
    index='userId',
    columns='title',
    values='rating'
)

# Preencher vazios
matriz = matriz.fillna(0)

# =========================
# SIMILARIDADE
# =========================

similaridade = cosine_similarity(matriz)

similaridade_df = pd.DataFrame(
    similaridade,
    index=matriz.index,
    columns=matriz.index
)

# =========================
# USUÁRIO ESCOLHIDO
# =========================

usuario = 1

# =========================
# TOP USUÁRIOS PARECIDOS
# =========================

usuarios_parecidos = similaridade_df[usuario].sort_values(
    ascending=False
)

# Remove o próprio usuário
usuarios_parecidos = usuarios_parecidos.drop(usuario)

# Pegar TOP 10 similares
top_usuarios = usuarios_parecidos.head(10)

print("TOP usuários similares:")
print(top_usuarios)

# =========================
# FILMES JÁ ASSISTIDOS
# =========================

filmes_assistidos = matriz.loc[usuario]
filmes_assistidos = filmes_assistidos[
    filmes_assistidos > 0
].index.tolist()

# =========================
# CALCULAR RECOMENDAÇÕES
# =========================

recomendacoes = {}

for usuario_similar, score_similaridade in top_usuarios.items():

    filmes_usuario = matriz.loc[usuario_similar]

    filmes_usuario = filmes_usuario[filmes_usuario > 0]

    for filme, nota in filmes_usuario.items():

        # Ignora filmes já vistos
        if filme in filmes_assistidos:
            continue

        # Fórmula de pontuação ponderada
        score = nota * score_similaridade

        if filme not in recomendacoes:
            recomendacoes[filme] = score
        else:
            recomendacoes[filme] += score

# =========================
# RANKING FINAL
# =========================

recomendacoes_df = pd.DataFrame(
    recomendacoes.items(),
    columns=['Filme', 'Pontuacao']
)

recomendacoes_df = recomendacoes_df.sort_values(
    by='Pontuacao',
    ascending=False
)

# =========================
# RESULTADO FINAL
# =========================

print("\nRECOMENDAÇÕES PARA O USUÁRIO", usuario)
print(recomendacoes_df.head(10))