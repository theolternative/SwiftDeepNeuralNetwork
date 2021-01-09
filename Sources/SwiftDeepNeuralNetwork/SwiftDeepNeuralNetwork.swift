import SwiftMatrix
import Foundation

public enum Initialization {
    case zeros
    case random
    case xavier
    case he
    case custom
}

public enum Optimization {
    case gradientdescent
    case momentum
    case adam
}

public enum Batch {
    case batch              // all examples altogether
    case minibatch          // mini_batch_size examples at a time
    case stochastic         // 1 example at a time
}
public class DeepNeuralNetwork {
    public var layer_dims : [Int]
    public var X : Matrix
    public var Y : Matrix
    public var W : [ Int : Matrix ] = [:]
    public var b : [ Int : Matrix ] = [:]
    
    public var learning_rate : Double = 0.0075
    public var num_iterations : Int = 2500
    public var weigth_init_type : Initialization = .random
    public var optimization_type : Optimization = .gradientdescent
    public var batch_type : Batch = .batch
    
    public var custom_weight_factor : Double = 1.0
    public var λ : Double = 0.0 // Regularization factor
    public var β : Double = 0.1 // Momentum optmization factor
    public var β1 : Double = 0.9 // Adam optmization factor
    public var β2 : Double = 0.999 // Adam optmization factor
    public var keep_prob : Double = 1.0 // Dropout keep propbability factor
    public var mini_batch_size : Int = 64

    private var keep_prob_weigths_cache : [Int:Matrix] = [:]
    private var vdW : [ Int : Matrix ] = [:]
    private var vdb : [ Int : Matrix ] = [:]
    private var sdW : [ Int : Matrix ] = [:]
    private var sdb : [ Int : Matrix ] = [:]
    private var t : Double = 0 // Adam optimization factor
    private var columnsIndices : [Int] = [] // Random mini batches
    
    public init( layerDimensions: [Int], X : Matrix, Y : Matrix ) {
        layer_dims = layerDimensions
        self.X = X
        self.Y = Y
    }
    
    func initialize_parameters() {
        let L = layer_dims.count
        
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
            b[l] = Matrix(rows: layer_dims[l], columns: 1, repeatedValue: 0.0)
        }
        
        switch( optimization_type ) {
        case .momentum:
            initialize_momentum_parameters()
            break
        case .adam:
            initialize_adam_parameters()
            break
        default:
            break
        }
    }
    
    func initialize_momentum_parameters() {
        let L = layer_dims.count

        for l in 1..<L {
            vdW[l] = Matrix(rows: layer_dims[l], columns: layer_dims[l-1], repeatedValue: 0.0)
            vdb[l] = Matrix(rows: layer_dims[l], columns: 1, repeatedValue: 0.0)
        }
    }

    func initialize_adam_parameters() {
        let L = layer_dims.count

        for l in 1..<L {
            vdW[l] = Matrix(rows: layer_dims[l], columns: layer_dims[l-1], repeatedValue: 0.0)
            vdb[l] = Matrix(rows: layer_dims[l], columns: 1, repeatedValue: 0.0)
            sdW[l] = Matrix(rows: layer_dims[l], columns: layer_dims[l-1], repeatedValue: 0.0)
            sdb[l] = Matrix(rows: layer_dims[l], columns: 1, repeatedValue: 0.0)
        }
    }

    public func random_mini_batches() -> [(Matrix, Matrix)]{
        let m = X.columns
        let num_complete_minibatches = (m/mini_batch_size)
        var mini_batches : [(Matrix, Matrix)] = []
        var index = 0
        columnsIndices = Array(0..<m).shuffled()
        for _ in 0..<num_complete_minibatches {
            var mini_batch_X = Matrix(rows: X.rows, columns: mini_batch_size, repeatedValue: 0.0)
            var mini_batch_Y = Matrix(rows: Y.rows, columns: mini_batch_size, repeatedValue: 0.0)
            for l in 0..<mini_batch_size {
                mini_batch_X[.all, l] = X[.all, columnsIndices[index] ]
                mini_batch_Y[.all, l] = Y[.all, columnsIndices[index] ]
                index += 1
            }
            let mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        }
        if m % mini_batch_size != 0 {
            let remainder = m - num_complete_minibatches * mini_batch_size
            var mini_batch_X = Matrix(rows: X.rows, columns: remainder, repeatedValue: 0.0)
            var mini_batch_Y = Matrix(rows: Y.rows, columns: remainder, repeatedValue: 0.0)
            for l in 0..<remainder {
                mini_batch_X[.all, l] = X[.all, columnsIndices[index] ]
                mini_batch_Y[.all, l] = Y[.all, columnsIndices[index] ]
                index += 1
            }
            let mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        }
        return mini_batches
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
            if( keep_prob < 1.0 ) {
                let probMatrix = Matrix.random(rows: A.rows, columns: A.columns, in: 0...1)
                let propMatrixWeights = probMatrix < keep_prob
                A *= propMatrixWeights
                A /= keep_prob
                keep_prob_weigths_cache[l] = propMatrixWeights
            }
            caches.append(cache)
        }
        ( AL, cache ) = linear_activation_forward(A, W[L]!, b[L]!, "sigmoid")

        caches.append(cache)
        
        return ( AL, caches )
    }
    
    func compute_cost( _ Yhat : Matrix, _ Y : Matrix ) -> Double {
        let m = Double(Yhat.columns)
        
        var L2_regularization_cost : Double = 0
        if( λ>0.0 ) {
            // Compute L2 Regularization
            let L = layer_dims.count
            for l in 1..<L {
                L2_regularization_cost += (Σ(W[l]!*W[l]!, .both))[0,0]
            }
            L2_regularization_cost *= (λ / (2 * m))
        }
        let cost = (-(Y ° log(Yhat′) + (1-Y) ° log( 1 - Yhat′) )/m)[0,0] + (λ>0.0 ? L2_regularization_cost : 0.0)

        return cost
    }
    
    func linear_backward( _ dZ : Matrix , _ linear_cache : (Matrix, Matrix, Matrix) ) -> (Matrix, Matrix, Matrix) {
        let (A_prev, W, _) = linear_cache
        let m = A_prev.columns
        let dW = λ>0.0 ? (dZ ° A_prev′)/m + W*λ/m : (dZ ° A_prev′)/m
        let db =  Matrix(columnVector: Σ(dZ, .row)/m, columns: 1)
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
            if( keep_prob < 1.0 ) {
                dA[l + 1] = dA[l + 1]! * keep_prob_weigths_cache[l+1]!
                dA[l + 1] = dA[l + 1]! / keep_prob
            }
            let (dA_prev_temp, dW_temp, db_temp) = linear_activation_backward(dA[l + 1]!, current_cache, "relu")
            dA[l] = dA_prev_temp
            dW[l + 1] = dW_temp
            db[l + 1] = db_temp
        }
        return (dA, dW, db)
    }
    func sigmoid( _ Z : Matrix ) -> ( Matrix, Matrix) {
        let A =  minel(1.0-1e-16,maxel(1e-16,1.0/(1.0+exp(-Z))))
        return (A, Z)
    }

    func update_parameters( _ parameters: ( [Int: Matrix], [Int: Matrix] ), _ grads : ( [Int: Matrix], [Int: Matrix], [Int: Matrix]) , _ learning_rate : Double ) -> ( [Int: Matrix], [Int: Matrix] ) {
        let L = layer_dims.count-1
        var (W_tmp, b_tmp) = parameters
        let (_, dW, db) = grads
        let epsilon = 1e-8
        
        switch( optimization_type ) {
        case .gradientdescent:
            for l in 0..<L {
                W_tmp[l+1] = W_tmp[l+1]! - learning_rate * dW[l+1]!
                b_tmp[l+1] = b_tmp[l+1]! - learning_rate * db[l+1]!
            }
            break
        case .momentum:
            for l in 0..<L {
                vdW[l+1] = β * vdW[l+1]! + (1-β) * dW[l+1]!
                vdb[l+1] = β * vdb[l+1]! + (1-β) * db[l+1]!
                W_tmp[l+1] = W_tmp[l+1]! - learning_rate * vdW[l+1]!
                b_tmp[l+1] = b_tmp[l+1]! - learning_rate * vdb[l+1]!
            }
            break
        case .adam:
            
            var vdW_corrected : [ Int : Matrix ] = [:]
            var vdb_corrected : [ Int : Matrix ] = [:]
            var sdW_corrected : [ Int : Matrix ] = [:]
            var sdb_corrected : [ Int : Matrix ] = [:]
            
            t = t + 1
            for l in 0..<L {
                vdW[l+1] = β1 * vdW[l+1]! + (1-β1) * dW[l+1]!
                vdb[l+1] = β1 * vdb[l+1]! + (1-β1) * db[l+1]!
                vdW_corrected[l+1] = vdW[l+1]! / ( 1.0 - pow(β1,t) )
                vdb_corrected[l+1] = vdb[l+1]! / ( 1.0 - pow(β1,t) )

                sdW[l+1] = β2 * sdW[l+1]! + (1-β2) * dW[l+1]! * dW[l+1]!
                sdb[l+1] = β1 * sdb[l+1]! + (1-β2) * db[l+1]! * db[l+1]!
                sdW_corrected[l+1] = sdW[l+1]! / ( 1.0 - pow(β2,t) )
                sdb_corrected[l+1] = sdb[l+1]! / ( 1.0 - pow(β2,t) )

                W_tmp[l+1] = W_tmp[l+1]! - learning_rate * vdW_corrected[l+1]! / (√(sdW_corrected[l+1]!)+epsilon)
                b_tmp[l+1] = b_tmp[l+1]! - learning_rate * vdb_corrected[l+1]! / (√(sdb_corrected[l+1]!)+epsilon)
            }
            break
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
        let dZ = (Z>=0.0)*dA
        
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
            let cost = compute_cost(A[2]!, Y)
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
        var Ypred = Matrix(rows:Y.rows, columns: Y.columns, repeatedValue: 0.0)
        switch( batch_type ) {
        case .batch:
            for i in 0..<num_iterations {
                let (AL, caches) = L_model_forward(X)
                let cost = compute_cost(AL, Y)
                if( cost.isNaN ) {
                    break
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
            break
        case .minibatch:
            let m = X.columns
            let mini_batches = random_mini_batches()
            for i in 0..<num_iterations {
                var cost = 0.0
                var Ypred_index = 0
                for mini_batch in mini_batches {
                    let (minibatch_X, minibatch_Y) = mini_batch
                    let (AL, caches) = L_model_forward(minibatch_X)
                    cost += (compute_cost(AL, minibatch_Y)*Double(AL.columns))
                    if( cost.isNaN ) {
                        break
                    }
                    let (dA, dW, db) = L_model_backward(AL, minibatch_Y, caches )
                    (W, b) = update_parameters( (W, b), (dA, dW, db), learning_rate)
                    for k in 0..<AL.columns {
                        Ypred[.all, Ypred_index] = AL[.all, k]
                        Ypred_index += 1
                    }
                }
                cost /= Double(m)
                if( i % 100 == 0 ) {
                    costs[i]=cost
                    print(String(format: "Cost after iteration %d : %.16f", i, cost))
                } else if( i==num_iterations-1) {
                    costs[i]=cost
                }
            }
            break
        case .stochastic:
            let m = X.columns
            for i in 0..<num_iterations {
                var cost = 0.0
                for j in 0..<m {
                    let (AL, caches) = L_model_forward(X[.all, j])
                    cost += compute_cost(AL, Y[.all, j])
                    if( cost.isNaN ) {
                        break
                    }
                    let (dA, dW, db) = L_model_backward(AL, Y[.all, j], caches )
                    (W, b) = update_parameters( (W, b), (dA, dW, db), learning_rate)
                    Ypred[.all, j] = AL
                }
                cost /= Double(m)
                if( i % 100 == 0 ) {
                    costs[i]=cost
                    print(String(format: "Cost after iteration %d : %.16f", i, cost))
                } else if( i==num_iterations-1) {
                    costs[i]=cost
                }
            }
            break
        }
        
        // Accuracy
        var correct = 0.0
        var accuracy = 0.0
        if( batch_type == .minibatch ) {
            for j in 0..<Y.columns {
                let yhat = (Ypred[0,j]<0.5 ? 0.0 : 1.0)
                if( yhat == Y[0,columnsIndices[j]] ) { correct += 1.0 }
            }
            accuracy = correct / Double(Ypred.columns)
        } else {
            var correct = 0.0
            for j in 0..<Y.columns {
                let yhat = (Ypred[0,j]<0.5 ? 0.0 : 1.0)
                if( yhat == Y[0,j] ) { correct += 1.0 }
            }
            accuracy = correct / Double(Ypred.columns)
        }
        return (accuracy, costs)
    }

    public func L_layer_model_predict( _ X_test : Matrix, _ Y_test : Matrix) -> Double {
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

