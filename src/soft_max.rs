use ndarray::{Array2, Axis, IxDyn};
use crate::operations::{Compute, Input, Output};

#[derive(Clone, Debug)]
pub struct SoftMax;

impl SoftMax {

    pub fn new() -> SoftMax {
        SoftMax
    }


    pub fn parse_from_proto_node() -> SoftMax {
        SoftMax
    }
    //attribute axis not used since its value is always = 1; no need for parsing whatsoever then
}

impl Compute for SoftMax {
    fn compute(&mut self, input: Input) -> Output {

        let matrix: Array2<f32> = match input {
            Input::TensorD(array) => array.into_dimensionality().unwrap(),
            _ => panic!("wrong input type")
        }; //Questa parte del codice gestisce il passaggio dalla variante Input al tipo Array2<f32>
        /*
        PREVIOUS VERSION
        let max_value = matrix.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let subtracted = matrix - max_value;

        let exp_values = subtracted.mapv(|x| x.exp());

        let sum_exp = exp_values.sum();

        let softmax_values = exp_values / sum_exp;
        */
        let max_values = matrix.fold_axis(Axis(1),f32::NEG_INFINITY,
                                          |a, b| a.max(b.clone()));
        //Questa linea calcola il massimo valore in ciascuna riga della matrice matrix e restituisce un nuovo vettore max_values contenente i massimi valori.

        let len = max_values.len();
        let subtracted = matrix - max_values.into_shape((len, 1)).unwrap();
        //Qui vengono sottratti i massimi valori calcolati in precedenza da ciascun elemento della matrice matrix

        let exp_values = subtracted.mapv(|x| x.exp());
        //: Questa riga calcola il valore esponenziale di ciascun elemento della matrice subtracted.

        let sum_exp = exp_values.fold_axis(Axis(1), 0.0, |a, b| a + b);
        // Questa linea calcola la somma degli elementi esponenziali in ciascuna riga della matrice exp_values.

        let len = sum_exp.len();

        let softmax_values: Array2<f32> = exp_values / sum_exp.into_shape((len, 1)).unwrap();
        //Qui, viene calcolato il softmax dividendo ogni elemento dell'exp_values per la somma degli elementi sum_exp. Questo restituisce una matrice di valori softmax.
        //In questa riga, exp_values è una matrice di valori esponenziali di dimensioni (n, m) e sum_exp è un vettore di somme di valori esponenziali di dimensione (n,). Il broadcasting avviene quando si esegue l'operazione di divisione tra exp_values e sum_exp.
        //
        // Nel broadcasting, Rust allinea automaticamente le dimensioni degli array in modo che
        // l'operazione possa essere eseguita senza errori. In questo caso, sum_exp è di dimensione (n,),
        // ma viene allineato in modo che possa essere diviso per exp_values che è di dimensione (n, m).
        // Il risultato di questa operazione di divisione sarà una matrice di dimensione (n, m) in
        // cui ogni elemento della riga i è stato diviso per sum_exp[i].

        let out_len  = Vec::from(softmax_values.shape());
        // Questa linea estrae le dimensioni della matrice softmax_values e le converte in un vettore.

        return Output::TensorD(softmax_values.into_shape(IxDyn(&out_len)).unwrap());

    }

    fn op_type(&self) -> &'static str {
        return "SoftMax";
    }
}


/*
Prendiamo un input che può essere un insieme di numeri in una matrice con molte dimensioni.

L'operazione SoftMax trasforma questo input in una matrice 2D (una matrice con due dimensioni)
in cui la prima dimensione è tipicamente la dimensione del batch (spesso indicata come "N"),
e la seconda dimensione contiene i valori dopo la trasformazione SoftMax (spesso indicata come "D").

L'asse lungo il quale facciamo questa trasformazione può essere specificato tramite il parametro
 chiamato "axis." Di solito, l'asse 0 rappresenta la dimensione del batch,
 ma possiamo specificare altri assi se necessario. Un valore negativo per "axis" indica che
 dovremmo contare le dimensioni a partire dalla fine (dalla fine intesa come dall'ultima dimensione
 dell'input): Ad esempio, se il valore di "axis" è -1, significa che SoftMax verrà applicato
 all'ultima dimensione dell'input, se il valore è -2, verrà applicato alla penultima dimensione,
 e così via.

 se hai un input di dimensione 2x3x4x5 e axis=1 vuol dire che la dimensione di input da considerare
 sarà 2x(3x4x5) = 2x60 dove tutti gli elementi presenti sulle dimensioni 3x4x5 vengono riallineati
 su un'unica dimensione -> per capire questo concetto devi capire l'indicizzazione, ovvero prendi
 tutti gli elementi sulla dimensione [0]x3x4x5 cioè TUTTI gli elementi che stanno sull'indice 0
 della prima dimensione e metti questi 60 elementi dentro alle posizioni del nuovo vettore di input
 dato da [0]x60 e così via. Una volta fatto questo applichi la softmax separatamente per ognuna delle
 due dimensioni e il tensore di output avrà effettivamente due dimensioni.

 Di fatto, puoi lavorare considerando sempre axis=1 perchè è l'unico caso che si presenta nelle nostre
 reti e quando fai la match per la dimensione del tensore puoi considerare tensori di dimensione massima = 4
 oppure in questo caso puoi considerare anche solo tensori di dimensione = 2 visto che è l'unico caso presente
 nelle nostre reti
 */
