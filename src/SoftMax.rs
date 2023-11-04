
use ndarray::{Array, array, Array1, Array2, ArrayD, Axis, Ix1, Ix2};
use ndarray::Dim;
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct SoftMax;

impl SoftMax {
    pub fn new() -> SoftMax {
        SoftMax
    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> SoftMax {
        SoftMax
    }
}

impl Compute for SoftMax {
    fn compute(&mut self, input: Input) -> Output {

        let matrix: Array2<f32> = match input {
            Input::TensorD(array) => array.into_dimensionality().unwrap(),
            _ => panic!("wrong input type")
        };


        //let matrix = Array2::from_shape_vec((1, data.len()), data.to_vec()).unwrap();

        let max_value = matrix.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let subtracted = matrix - max_value;

        let exp_values = subtracted.mapv(|x| x.exp());

        let sum_exp = exp_values.sum();

        let softmax_values = exp_values / sum_exp;

        return match softmax_values.shape().len() {
            2 => Output::Tensor2(softmax_values.into_dimensionality::<Ix2>().unwrap()),
            _ => panic!("Wrong shape dim reshape")
        }

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
