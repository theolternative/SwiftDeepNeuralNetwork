import SwiftMatrix

public enum WeightsInitialization {
    case zeros
    case random
    case xavier
    case he
    case custom
}

public class DeepNeuralNetwork {
    public var layer_dims : [Int]
    public var X : Matrix
    public var Y : Matrix
    public var W : [ Int : Matrix ] = [:]
    public var b : [ Int : Matrix ] = [:]
    
    public var learning_rate : Double = 0.0075
    public var num_iterations : Int = 2500
    public var weigth_init_type : WeightsInitialization = .random
    public var custom_weight_factor : Double = 1.0
    public var λ : Double = 0.0 // Regularization factor
    
    public init( layerDimensions: [Int], X : Matrix, Y : Matrix ) {
        layer_dims = layerDimensions
        self.X = X
        self.Y = Y
    }
    
    func initialize_parameters() {
        let L = layer_dims.count
        let m = X.columns
        
        for l in 1..<L {
            switch( weigth_init_type ) {
            case .zeros:
                W[l] = Matrix(rows: layer_dims[l], columns: layer_dims[l-1], repeatedValue: 0.0)
                break;
            case .xavier:
                W[l] = Matrix.random(rows: layer_dims[l], columns: layer_dims[l-1], in: -1...1) * ((1.0/Double(layer_dims[l-1])).squareRoot())
                break;
            case .he:
                W[l] = Matrix.random(rows: layer_dims[l], columns: layer_dims[l-1], in: -1...1) * ((2.0/Double(layer_dims[l-1])).squareRoot())
                break;
            case .custom:
                W[l] = custom_weight_factor*Matrix.random(rows: layer_dims[l], columns: layer_dims[l-1], in: -1...1)
                break;
            case .random:
                W[l] = Matrix.random(rows: layer_dims[l], columns: layer_dims[l-1], in: -1...1)
                break;
            }
            b[l] = Matrix(rows: layer_dims[l], columns: m, repeatedValue: 0.0)
        }
    }
    
    func linear_forward( _ A : Matrix, _ W: Matrix, _ b: Matrix) -> ( Matrix, (Matrix, Matrix, Matrix) ) {
        let Z = W°A+b
        let cache = (A, W, b)
        
        return( Z: Z, cache: cache )
    }

    func linear_activation_forward( _ A_prev : Matrix, _ W: Matrix, _ b: Matrix, _ activation: String) -> (Matrix, (( Matrix, Matrix, Matrix ), Matrix)) {
        let (Z, linear_cache) = linear_forward(A_prev, W, b)
        let (A, activation_cache) = activation=="sigmoid" ? sigmoid(Z): relu(Z)
        let cache = (linear_cache, activation_cache)
        
        return( A: A, cache: cache )
    }
    
    func L_model_forward( _ X : Matrix ) -> (Matrix, [(( Matrix, Matrix, Matrix ), Matrix)] ) {
        var A = X
        var caches : [(( Matrix, Matrix, Matrix ), Matrix)] = []
        var cache : (( Matrix, Matrix, Matrix ), Matrix)
        var AL : Matrix
        let L = layer_dims.count-1 // Number of parameters which is number of layers - 1
        for l in 1..<L {
            let A_prev = A
            ( A, cache ) = linear_activation_forward(A_prev, W[l]!, b[l]!, "relu")
            caches.append(cache)
        }
        ( AL, cache ) = linear_activation_forward(A, W[L]!, b[L]!, "sigmoid")
        caches.append(cache)
        
        return ( AL, caches )
    }
    
    func compute_cost( _ Yhat : Matrix ) -> Double {
        let m = Yhat.columns

        return (-(Y ° log(Yhat′) + (1-Y) ° log( 1 - Yhat′) )/m)[0,0]
    }
    
    func linear_backward( _ dZ : Matrix , _ linear_cache : (Matrix, Matrix, Matrix) ) -> (Matrix, Matrix, Matrix) {
        let (A_prev, W, _) = linear_cache
        let m = A_prev.columns
        let dW = (dZ ° A_prev′)/m
        let db =  Matrix(columnVector: Σ(dZ, .row)/m, columns: m)
        let dA_prev = W′ ° dZ
        
        return (dA_prev, dW, db)
    }
    
    func linear_activation_backward( _ dA : Matrix, _ linear_activation_cache : ((Matrix, Matrix, Matrix), Matrix) , _ activation : String) -> (Matrix, Matrix, Matrix) {
        let (linear_cache, activation_cache) = linear_activation_cache
        let dZ = (activation == "sigmoid" ? sigmoid_backward(dA, activation_cache) : relu_backward(dA, activation_cache ))
        let (dA_prev, dW, db) = linear_backward(dZ, linear_cache)
        return (dA_prev, dW, db)
    }
    
    func L_model_backward( _ AL : Matrix, _ Y : Matrix, _ caches: [(( Matrix, Matrix, Matrix ), Matrix)] ) -> ( [Int: Matrix], [Int: Matrix], [Int: Matrix]) {
        let L = layer_dims.count-1
        var dA : [Int: Matrix] = [:]
        var dW : [Int: Matrix] = [:]
        var db : [Int: Matrix] = [:]
        
        let dAL = -((Y / AL) - ( (1-Y) / (1-AL) ))
        var current_cache = caches[L-1]
        (dA[L-1], dW[L], db[L]) = linear_activation_backward(dAL, current_cache, "sigmoid")
        for l in (0..<L-1).reversed() {
            current_cache = caches[l]
            let (dA_prev_temp, dW_temp, db_temp) = linear_activation_backward(dA[l + 1]!, current_cache, "relu")
            dA[l] = dA_prev_temp
            dW[l + 1] = dW_temp
            db[l + 1] = db_temp
        }
        return (dA, dW, db)
    }
    func sigmoid( _ Z : Matrix ) -> ( Matrix, Matrix) {
        let A = 1.0/(1.0+exp(-Z))
        return (A, Z)
    }

    func update_parameters( _ parameters: ( [Int: Matrix], [Int: Matrix] ), _ grads : ( [Int: Matrix], [Int: Matrix], [Int: Matrix]) , _ learning_rate : Double ) -> ( [Int: Matrix], [Int: Matrix] ) {
        let L = layer_dims.count-1
        var (W_tmp, b_tmp) = parameters
        let (_, dW, db) = grads
        for l in 0..<L {
            W_tmp[l+1] = W_tmp[l+1]! - learning_rate * dW[l+1]!
            b_tmp[l+1] = b_tmp[l+1]! - learning_rate * db[l+1]!
        }
        return (W_tmp, b_tmp)
    }
    func relu( _ Z : Matrix ) -> ( Matrix, Matrix) {
        let A = maxel(0.0, Z)
        return (A, Z)
    }
    
    func sigmoid_backward( _ dA : Matrix, _ activation_cache : Matrix ) -> Matrix {
        let Z = activation_cache
        
        let A = 1.0/(1.0+exp(-Z))
        let dZ = dA * A * (1-A)
        
        return dZ
    }
    
    func relu_backward( _ dA : Matrix, _ activation_cache : Matrix ) -> Matrix {
        let Z = activation_cache
        var dZ = dA
        for row in 0..<Z.rows {
            for column in 0..<Z.columns {
                dZ[row, column] = (Z[row, column]<0 ? 0 : dA[row, column])
            }
        }
        return dZ
    }
    
    public func two_layer_model_train(_ W1t : Matrix?, _ W2t : Matrix? ) -> (Double, [Int:Double]) {
        var A : [Int:Matrix] = [:]
        var dA : [Int:Matrix] = [:]
        var dW : [Int:Matrix] = [:]
        var db : [Int:Matrix] = [:]
        var cache : [Int:(( Matrix, Matrix, Matrix ), Matrix)] = [:]
        var costs : [Int:Double] = [:]
        
        initialize_parameters()
        if( W1t != nil ) {
            W[1]=W1t
        }
        if( W2t != nil ) {
            W[2]=W2t
        }
        for i in 0..<num_iterations {
            (A[1], cache[1]) = linear_activation_forward(X, W[1]!, b[1]!, "relu")
            (A[2], cache[2]) = linear_activation_forward(A[1]!, W[2]!, b[2]!, "sigmoid")
            let cost = compute_cost(A[2]!)
            dA[2] = -((Y / A[2]!) - ((1 - Y) / (1 - A[2]!)))
            (dA[1], dW[2], db[2]) = linear_activation_backward(dA[2]!, cache[2]!, "sigmoid")
            (dA[0], dW[1], db[1]) = linear_activation_backward(dA[1]!, cache[1]!, "relu")
            
            (W, b) = update_parameters( (W, b), (dA, dW, db), learning_rate)
            if( i % 100 == 0 ) {
                costs[i]=cost
                print(String(format: "Cost after iteration %d : %.16f", i, cost))
            } else if( i==num_iterations-1) {
                costs[i]=cost
            }
        }
        
        // Accuracy
        let Ypred = A[2]!
        var correct = 0.0
        for j in 0..<Ypred.columns {
            let yhat = (Ypred[0,j]<0.5 ? 0.0 : 1.0)
            if( yhat == Y[0,j] ) { correct += 1.0 }
        }
        let accuracy = correct / Double(Ypred.columns)
        return (accuracy, costs)
    }
    
    public func two_layer_model_predict( _ X_test : Matrix, _ Y_test : Matrix) -> Double {
        var A : [Int:Matrix] = [:]
        var cache : [Int:(( Matrix, Matrix, Matrix ), Matrix)] = [:]
        // Adapt b size
        for (key, _) in b {
            var btmp_grid = [Double](repeating: 0.0, count: b[key]!.rows * X_test.columns)
            for i in 0..<b[key]!.rows {
                for j in 0..<X_test.columns {
                    btmp_grid[i*X_test.columns+j] = b[key]![i,0]
                }
            }
            b[key] = Matrix(rows: b[key]!.rows, columns: X_test.columns, values: btmp_grid)
        }

        (A[1], cache[1]) = linear_activation_forward(X_test, W[1]!, b[1]!, "relu")
        (A[2], cache[2]) = linear_activation_forward(A[1]!, W[2]!, b[2]!, "sigmoid")
        
        // Accuracy
        let Ypred = A[2]!
        var correct = 0.0
        for j in 0..<Ypred.columns {
            let yhat = (Ypred[0,j]<0.5 ? 0.0 : 1.0)
            if( yhat == Y_test[0,j] ) { correct += 1.0 }
        }
        let accuracy = correct / Double(Ypred.columns)
        return accuracy

    }
    
    public func L_layer_model_train( _ Wt : [Int:Matrix]? ) -> (Double, [Int:Double]) {
        var costs : [Int:Double] = [:]
        initialize_parameters()
        if( Wt != nil ) {
            W = Wt!
        }
        var Ypred = Matrix(rows:0, columns: 0, repeatedValue: 0.0)
        for i in 0..<num_iterations {
            let (AL, caches) = L_model_forward(X)
            let cost = compute_cost(AL)
            if( cost.isNaN ) {
                break;
            }
            let (dA, dW, db) = L_model_backward(AL, Y, caches )
            (W, b) = update_parameters( (W, b), (dA, dW, db), learning_rate)
            if( i % 100 == 0 ) {
                costs[i]=cost
                print(String(format: "Cost after iteration %d : %.16f", i, cost))
            } else if( i==num_iterations-1) {
                costs[i]=cost
            }
            Ypred = AL
        }
        
        // Accuracy
        var correct = 0.0
        for j in 0..<Ypred.columns {
            let yhat = (Ypred[0,j]<0.5 ? 0.0 : 1.0)
            //print("\(Ypred[0,j]) -> \(yhat) should be \(Y[0,j])")
            if( yhat == Y[0,j] ) { correct += 1.0 }
        }
        let accuracy = correct / Double(Ypred.columns)
        return (accuracy, costs)
    }

    public func L_layer_model_predict( _ X_test : Matrix, _ Y_test : Matrix) -> Double {
        // Adapt b size
        for (key, _) in b {
            var btmp_grid = [Double](repeating: 0.0, count: b[key]!.rows * X_test.columns)
            for i in 0..<b[key]!.rows {
                for j in 0..<X_test.columns {
                    btmp_grid[i*X_test.columns+j] = b[key]![i,0]
                }
            }
            b[key] = Matrix(rows: b[key]!.rows, columns: X_test.columns, values: btmp_grid)
        }
        let (AL, _) = L_model_forward(X_test)
        
        // Accuracy
        let Ypred = AL
        var correct = 0.0
        for j in 0..<Ypred.columns {
            let yhat = (Ypred[0,j]<0.5 ? 0.0 : 1.0)
            if( yhat == Y_test[0,j] ) { correct += 1.0 }
        }
        let accuracy = correct / Double(Ypred.columns)
        return accuracy

    }

    public static func csv(fileUrl: String) -> Matrix {
        var rowsCount = 0, columnsCount = 0
        var values: [Double] = []
        do {
            let data = try String(contentsOfFile: fileUrl, encoding: String.Encoding.utf8)
            let rows = data.components(separatedBy: "\r\n")
            rowsCount = rows.count
            for row in rows {
                let columns = row.components(separatedBy: ";")
                columnsCount = columns.count
                for column in columns {
                    values.append(Double(column)!)
                }
            }
        } catch {
            print("I/O Error")
        }
        return Matrix(rows: rowsCount, columns: columnsCount, values: values)
    }
}

