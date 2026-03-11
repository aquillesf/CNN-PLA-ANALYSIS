import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


def parse_pla(filepath):
    X_raw = []
    y_raw = []
    n_inputs = None
    n_outputs = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('.i '):       
                n_inputs = int(line.split()[1])
            elif line.startswith('.o '):      
                n_outputs = int(line.split()[1])
            elif line.startswith('.'):        
                continue
            elif line == '':
                continue
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                entrada = parts[0]
                saida   = parts[1]
                if len(entrada) == n_inputs:
                    X_raw.append(entrada)
                    y_raw.append(saida)

    print(f"Entradas:  {n_inputs}")
    print(f"Saidas:    {n_outputs}")
    print(f"Linhas no PLA: {len(X_raw)}")
    np.random.seed(42)
    X_list, y_list = [], []
    pos_set = set()

    for entrada, saida in zip(X_raw, y_raw):
        row_x = []
        for ch in entrada:
            if ch == '0':
                row_x.append(0)
            elif ch == '1':
                row_x.append(1)
            else:  
                row_x.append(np.random.randint(0, 2))

        #saida ~ trata como 0
        row_y = []
        for ch in saida:
            if ch == '1':
                row_y.append(1)
            else: 
                row_y.append(0)

        X_list.append(row_x)
        y_list.append(row_y)
        pos_set.add(tuple(row_x))

    n_pos = len(X_list)
    neg_count = 0
    tentativas = 0
    while neg_count < n_pos and tentativas < n_pos * 200:
        row = list(np.random.randint(0, 2, size=n_inputs))
        if tuple(row) not in pos_set:
            X_list.append(row)
            y_list.append([0] * n_outputs)
            neg_count += 1
        tentativas += 1

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    perm = np.random.permutation(len(X))
    return X[perm], y[perm], n_inputs, n_outputs


def build_model(n_inputs, n_outputs):
    model = keras.Sequential([
        layers.Input(shape=(n_inputs, 1)),
        layers.Conv1D(16, kernel_size=3, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_outputs, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":

    ARQUIVO = "duke2.pla"

    print(f"Carregando {ARQUIVO}...")
    X, y, n_inputs, n_outputs = parse_pla(ARQUIVO)
    print(f"Total de amostras: {len(X)}\n")

    test_size = 0.2 if len(X) >= 50 else 0.15

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    #reshape pra CNN 1D
    X_train = X_train.reshape(-1, n_inputs, 1)
    X_test  = X_test.reshape(-1, n_inputs, 1)

    model = build_model(n_inputs, n_outputs)
    model.summary()
    epocas = 100 if len(X) < 200 else 30

    print(f"\nTreinando por {epocas} epocas...")
    model.fit(
        X_train, y_train,
        epochs=epocas,
        batch_size=16,
        validation_split=0.1,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {loss:.4f}  |  Acuracia: {acc*100:.2f}%")

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    if n_outputs == 1:
        print(classification_report(y_test.astype(int), y_pred,
                                    target_names=['Saida 0', 'Saida 1']))
        print("Matriz de Confusao:")
        print(confusion_matrix(y_test.astype(int).flatten(), y_pred.flatten()))
    else:
        for i in range(n_outputs):
            acerto = np.mean(y_pred[:, i] == y_test[:, i].astype(int))
            print(f"  Saida {i+1}: acuracia = {acerto*100:.1f}%")
        total_possiveis = 2 ** n_inputs
        print(f"  Combinacoes possiveis: {total_possiveis}")

        entradas_no_arquivo = set()
        with open(ARQUIVO, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('.') or line == '':
                    continue
                parts = line.split()
                if len(parts) >= 2 and len(parts[0]) == n_inputs:
                    if '-' not in parts[0]:
                        entradas_no_arquivo.add(parts[0])

        print(f"  Combinacoes no arquivo: {len(entradas_no_arquivo)}")
        faltando = []
        for i in range(total_possiveis):
            bits = format(i, f'0{n_inputs}b')
            if bits not in entradas_no_arquivo:
                faltando.append(bits)

        print(f"  Combinacoes FALTANDO:   {len(faltando)}")

        if len(faltando) == 0:
            print("  Nenhuma combinacao faltando")
        else:
            X_faltando = np.array(
                [[int(b) for b in bits] for bits in faltando],
                dtype=np.float32
            ).reshape(-1, n_inputs, 1)
            previsoes = model.predict(X_faltando, verbose=0)
            previsoes_bin = (previsoes >= 0.5).astype(int)
            nome_saida = ARQUIVO.replace('.pla', '_completo.pla')

            with open(nome_saida, 'w') as f:
                f.write(f".i {n_inputs}\n")
                f.write(f".o {n_outputs}\n")
                f.write(f".p {len(faltando)}\n")
                for bits, saida in zip(faltando, previsoes_bin):
                    saida_str = ''.join(str(b) for b in saida)
                    f.write(f"{bits} {saida_str}\n")
                f.write(".e\n")
            print(f"  Primeiras 10 combinacoes faltantes previstas:")
            for bits, saida in zip(faltando[:10], previsoes_bin[:10]):
                saida_str = ''.join(str(b) for b in saida)
                print(f"    {bits}  ->  {saida_str}")
            if len(faltando) > 10:
                print(f"    ... e mais {len(faltando) - 10} linhas no arquivo")
