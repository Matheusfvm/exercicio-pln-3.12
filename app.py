import pandas as pd

from preprocessing import preprocess_text
from vectorizer_bow import vectorizer
from modeling_lda import fit_lda_model

documents = [
    'Não consigo fazer acessar minha conta: ao inserir usuário e senha, recebo a mensagem Credenciais inválidas',
    'Solicito desbloqueio de conta: fui bloqueado após tentar acessar várias vezes o e-mail corporativo',
    'Preciso acessar os arquivos financeiro, mas o perfil não foi liberado (mensagem: Acesso negado)',
    'Não consigo acessar os documentos, aparece você não tem permissão para visualizar eles',
    'No serviço de contabilidade, não dá para lançar notas fiscais',
    'O site está lento, a barra de progresso fica parada',
    'Após atualizar o antivírus, alguns serviços não iniciam e exibem erro de licença',
    'Ao abrir o serviço ele fecha inesperadamente',
    'O serviço não converte corretamente o arquivo .docx em PDF, formatação fica toda desconfigurada',
    'O navegador abre normalmente, mas a página fica carregando indefinidamente',
    'O computador não reconhece meu áudio, aparece microfone não encontrado',
    'Meu notebook não liga: ao pressionar o botão de ligar, nenhuma luz acende e não há sinal de vida',
    'Teclado não funciona: várias teclas travam e, por vezes, não digitam nada',
    'O monitor está ligando e desligando intermitente, principalmente ao alternar janelas',
    'O projetor da sala de reuniões não liga, ao acionar o botão de ligar, só pisca a luz vermelha',
    'A ventoinha do servidor está fazendo barulho alto e a máquina reinicia sozinha às vezes'
]

preprocessed_documents = [preprocess_text(doc) for doc in documents]

bow, vocab = vectorizer(preprocessed_documents)

n_topics = 3

dt_matrix, topic_word_matrix = fit_lda_model(bow, n_topics)

n_top_words = 10

df_topic_distr = pd.DataFrame(
    dt_matrix,
    columns=[f'tópico_{i+1}' for i in range(n_topics)]
)
print('\n=== Distribuição de Tópicos por Documento ===')
print(df_topic_distr)
print('\n=== Top palavras em cada Tópico ===')
for topic_idx, topic_weights in enumerate(topic_word_matrix):
    top_indices = topic_weights.argsort()[::-1][:n_top_words]
    top_palavras = [vocab[i] for i in top_indices]
    top_pesos   = [topic_weights[i] for i in top_indices]
    print(f'Tópico {topic_idx+1}:')
    for palavra, peso in zip(top_palavras, top_pesos):
        print(f'   {palavra:<20} ({peso:.2f})')
    print()