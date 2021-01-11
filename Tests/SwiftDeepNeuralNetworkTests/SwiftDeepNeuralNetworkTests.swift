import XCTest
import Foundation
import SwiftMatrix

@testable import SwiftDeepNeuralNetwork

final class SwiftDeepNeuralNetworkTests: XCTestCase {
    func testLinearForward() {
        // Expected result Z: [[ 3.26295337 -1.23429987]]

        let A = Matrix([[ 1.62434536, -0.61175641],
                    [-0.52817175, -1.07296862],
                    [ 0.86540763, -2.3015387 ]])
        let W = Matrix([[ 1.74481176, -0.7612069,   0.3190391 ]])
        let b = Matrix([[-0.24937038]])

        let X = Matrix(rows:2, columns:2, repeatedValue: 0.0)
        let Y = Matrix(rows:2, columns:1, repeatedValue: 0.0)

        let dnn = DeepNeuralNetwork(layerDimensions: [2,3,1], X: X, Y:Y)
        let (Z, _) = dnn.linear_forward(A, W, b)

        XCTAssertEqual(Z[0,0], 3.26295337, accuracy: 1e-7)
        XCTAssertEqual(Z[0,1], -1.23429987, accuracy: 1e-7)
    }
    
    func testLinearActivationForward() {
        // Expected results
        //  With sigmoid: A    [[ 0.96890023 0.11013289]]
        //  With ReLU: A    [[ 3.43896131 0. ]]

        let A_prev = Matrix([[-0.41675785, -0.05626683],
                         [-2.1361961,   1.64027081],
                         [-1.79343559, -0.84174737]])
        let W = Matrix([[ 0.50288142, -1.24528809, -1.05795222]])
        let b =  Matrix([[-0.90900761]])

        let X = Matrix(rows:2, columns:2, repeatedValue: 0.0)
        let Y = Matrix(rows:2, columns:1, repeatedValue: 0.0)

        let dnn = DeepNeuralNetwork(layerDimensions: [2,3,1], X: X, Y:Y)
        let( As, _ ) = dnn.linear_activation_forward(A_prev, W, b, "sigmoid")
        
        XCTAssertEqual(As[0,0], 0.96890023, accuracy: 1e-7, "Sigmoid test failed")
        XCTAssertEqual(As[0,1], 0.11013289, accuracy: 1e-7, "Sigmoid test failed")

        let( Ar, _ ) = dnn.linear_activation_forward(A_prev, W, b, "relu")
        XCTAssertEqual(Ar[0,0], 3.43896131, accuracy: 1e-7, "Relu test failed")
        XCTAssertEqual(Ar[0,1], 0.0, accuracy: 1e-7, "Relu test failed")
    }

    func testLModelForward() {
        // Expected results
        // AL    [[ 0.03921668 0.70498921 0.19734387 0.04728177]]

        let X = Matrix([[-0.31178367,  0.72900392,  0.21782079, -0.8990918 ],
                    [-2.48678065, 0.91325152,  1.12706373, -1.51409323],
                    [ 1.63929108, -0.4298936,   2.63128056,  0.60182225],
                    [-0.33588161,  1.23773784,  0.11112817,  0.12915125],
                    [ 0.07612761, -0.15512816,  0.63422534,  0.810655  ]])
        let Y = Matrix(rows:5, columns:1, repeatedValue: 0.0)

        let dnn = DeepNeuralNetwork(layerDimensions: [5,4,3,1], X: X, Y:Y)
        dnn.W[1] = Matrix([[ 0.35480861,  1.81259031, -1.3564758 , -0.46363197,  0.82465384],
                       [-1.17643148,  1.56448966,  0.71270509, -0.1810066 ,  0.53419953],
                       [-0.58661296, -1.48185327,  0.85724762,  0.94309899,  0.11444143],
                       [-0.02195668, -2.12714455, -0.83440747, -0.46550831,  0.23371059]])
        dnn.b[1] = Matrix([[ 1.38503523],
                       [-0.51962709],
                       [-0.78015214],
                       [ 0.95560959]])
        dnn.W[2] = Matrix([[-0.12673638, -1.36861282,  1.21848065, -0.85750144],
                       [-0.56147088, -1.0335199 ,  0.35877096,  1.07368134],
                       [-0.37550472,  0.39636757, -0.47144628,  2.33660781]])
        dnn.b[2] = Matrix([[ 1.50278553],
                       [-0.59545972],
                       [ 0.52834106]])
        dnn.W[3] = Matrix( [[ 0.9398248 ,  0.42628539, -0.75815703]] )
        dnn.b[3] = Matrix( [[-0.16236698]])
        let( AL, _ ) = dnn.L_model_forward(X)
        XCTAssertEqual(AL[0,0], 0.03921668, accuracy: 1e-7 )
        XCTAssertEqual(AL[0,1], 0.70498921, accuracy: 1e-7 )
        XCTAssertEqual(AL[0,2], 0.19734387, accuracy: 1e-7 )
        XCTAssertEqual(AL[0,3], 0.04728177, accuracy: 1e-7 )
    }
    
    func testCostFunction() {
        // Expected result cost 0.2797765635793422
       let X = Matrix([[-0.31178367,  0.72900392,  0.21782079, -0.8990918 ],
                    [-2.48678065, 0.91325152,  1.12706373, -1.51409323],
                    [ 1.63929108, -0.4298936,   2.63128056,  0.60182225],
                    [-0.33588161,  1.23773784,  0.11112817,  0.12915125],
                    [ 0.07612761, -0.15512816,  0.63422534,  0.810655  ]])
        let Y = Matrix([[1, 1, 0]])

        let dnn = DeepNeuralNetwork(layerDimensions: [5,1], X: X, Y:Y)
        let AL = Matrix([[ 0.8,  0.9,  0.4]])
        let cost = dnn.compute_cost(AL, Y)
        XCTAssertEqual(cost, 0.2797765635793422, accuracy: 1e-7 )
    }
    
    func testLinearBackward() {
        // Expected results
        // dA_prev =
        //  [[-1.15171336  0.06718465 -0.3204696   2.09812712]
        //  [ 0.60345879 -3.72508701  5.81700741 -3.84326836]
        //  [-0.4319552  -1.30987417  1.72354705  0.05070578]
        //  [-0.38981415  0.60811244 -1.25938424  1.47191593]
        //  [-2.52214926  2.67882552 -0.67947465  1.48119548]]
        // dW =
        //  [[ 0.07313866 -0.0976715  -0.87585828  0.73763362  0.00785716]
        //  [ 0.85508818  0.37530413 -0.59912655  0.71278189 -0.58931808]
        //  [ 0.97913304 -0.24376494 -0.08839671  0.55151192 -0.10290907]]
        // db =
        //  [[-0.14713786]
        //  [-0.11313155]
        //  [-0.13209101]]

        let X = Matrix([[-0.31178367,  0.72900392,  0.21782079, -0.8990918 ],
                    [-2.48678065, 0.91325152,  1.12706373, -1.51409323],
                    [ 1.63929108, -0.4298936,   2.63128056,  0.60182225],
                    [-0.33588161,  1.23773784,  0.11112817,  0.12915125],
                    [ 0.07612761, -0.15512816,  0.63422534,  0.810655  ]])
        let Y = Matrix([[1, 1, 0]])

        let dnn = DeepNeuralNetwork(layerDimensions: [5,4,3,1], X: X, Y:Y)
        let dZ = Matrix([[ 1.62434536, -0.61175641, -0.52817175, -1.07296862],
                     [ 0.86540763, -2.3015387,   1.74481176, -0.7612069 ],
                     [ 0.3190391,  -0.24937038,  1.46210794, -2.06014071]])
        let linear_cache = (
                    // A_prev
                    Matrix([[-0.3224172 , -0.38405435,  1.13376944, -1.09989127],
                        [-0.17242821, -0.87785842,  0.04221375,  0.58281521],
                        [-1.10061918,  1.14472371,  0.90159072,  0.50249434],
                        [ 0.90085595, -0.68372786, -0.12289023, -0.93576943],
                        [-0.26788808,  0.53035547, -0.69166075, -0.39675353]]),
                     // W
                    Matrix([[-0.6871727 , -0.84520564, -0.67124613, -0.0126646 , -1.11731035],
                        [ 0.2344157 ,  1.65980218,  0.74204416, -0.19183555, -0.88762896],
                        [-0.74715829,  1.6924546 ,  0.05080775, -0.63699565,  0.19091548]]),
                    // b
                    Matrix([[ 2.10025514],
                        [ 0.12015895],
                        [ 0.61720311]]))
        let (dA, dW, db) = dnn.linear_backward(dZ, linear_cache)
        
        XCTAssertEqual(dA[0,0], -1.15171336, accuracy: 1e-7 )
        XCTAssertEqual(dA[0,1], 0.06718465, accuracy: 1e-7 )
        XCTAssertEqual(dA[1,0], 0.60345879, accuracy: 1e-7 )
        XCTAssertEqual(dA[3,2], -1.25938424, accuracy: 1e-7 )

        XCTAssertEqual(dW[0,0], 0.07313866, accuracy: 1e-7 )
        XCTAssertEqual(dW[0,1], -0.0976715, accuracy: 1e-7 )
        XCTAssertEqual(dW[1,0], 0.85508818, accuracy: 1e-7 )
        XCTAssertEqual(dW[2,1], -0.24376494, accuracy: 1e-7 )

        XCTAssertEqual(db[0,0], -0.14713786, accuracy: 1e-7 )
        XCTAssertEqual(db[1,0], -0.11313155, accuracy: 1e-7 )
        XCTAssertEqual(db[2,0], -0.13209101, accuracy: 1e-7 )
    }
    
    func testLinearActivationBackward() {
        /* Expected results
         
         Expected output with sigmoid:
         dA_prev    [[ 0.11017994 0.01105339] [ 0.09466817 0.00949723] [-0.05743092 -0.00576154]]
         dW    [[ 0.10266786 0.09778551 -0.01968084]]
         db    [[-0.05729622]]
         Expected output with relu:
         dA_prev    [[ 0.44090989 0. ] [ 0.37883606 0. ] [-0.2298228 0. ]]
         dW    [[ 0.44513824 0.37371418 -0.10478989]]
         db    [[-0.20837892]]
        */

        let dAL = Matrix([[-0.41675785, -0.05626683]])
        let ((A, W, b), Z) = ((Matrix([[-2.1361961 ,  1.64027081],
                                   [-1.79343559, -0.84174737],
                                   [ 0.50288142, -1.24528809]]),
                               Matrix([[-1.05795222, -0.90900761,  0.55145404]]),
                               Matrix([[ 2.29220801, 2.29220801]])),
                               Matrix([[ 0.04153939, -1.11792545]]))

        let X = Matrix([[-0.31178367,  0.72900392 ],
                    [-2.48678065, 0.91325152 ],
                    [ 1.63929108, -0.4298936 ],
                    [-0.33588161,  1.23773784 ],
                    [ 0.07612761, -0.15512816 ]])
        let Y = Matrix([[1, 1]])

        let dnn = DeepNeuralNetwork(layerDimensions: [5,1], X: X, Y:Y)
        let (dAprevs, dWs, dbs) = dnn.linear_activation_backward(dAL, ((A, W, b), Z), "sigmoid")
        
        XCTAssertEqual(dAprevs[0,0], 0.11017994, accuracy: 1e-7 )
        XCTAssertEqual(dAprevs[1,0], 0.09466817, accuracy: 1e-7 )
        XCTAssertEqual(dWs[0,0], 0.10266786, accuracy: 1e-7 )
        XCTAssertEqual(dWs[0,2], -0.01968084, accuracy: 1e-7 )
        XCTAssertEqual(dbs[0,0], -0.05729622, accuracy: 1e-7 )

        let (dAprevr, dWr, dbr) = dnn.linear_activation_backward(dAL, ((A, W, b), Z), "relu")
        XCTAssertEqual(dAprevr[0,0], 0.44090989, accuracy: 1e-7 )
        XCTAssertEqual(dAprevr[1,0], 0.37883606, accuracy: 1e-7 )
        XCTAssertEqual(dWr[0,0], 0.44513824, accuracy: 1e-7 )
        XCTAssertEqual(dWr[0,2], -0.10478989, accuracy: 1e-7 )
        XCTAssertEqual(dbr[0,0], -0.20837892, accuracy: 1e-7 )
    }
    
    func testLModelBackward() {
        // Expected results
        // dW1    [[ 0.41010002 0.07807203 0.13798444 0.10502167] [ 0. 0. 0. 0. ] [ 0.05283652 0.01005865 0.01777766 0.0135308 ]]
        // db1    [[-0.22007063] [ 0. ] [-0.02835349]]
        // dA1    [[ 0.12913162 -0.44014127] [-0.14175655 0.48317296] [ 0.01663708 -0.05670698]]
        let X = Matrix([[-0.31178367,  0.72900392 ],
                     [-2.48678065, 0.91325152 ],
                     [ 1.63929108, -0.4298936 ],
                     [-0.33588161,  1.23773784 ],
                     [ 0.07612761, -0.15512816 ]])
        let Y = Matrix([[1, 1]])
        let AL = Matrix([[ 1.78862847,  0.43650985]])
        let Y_assess = Matrix([[1, 0]])
        let caches = [((Matrix([[ 0.09649747, -1.8634927 ],
                               [-0.2773882 , -0.35475898],
                               [-0.08274148, -0.62700068],
                               [-0.04381817, -0.47721803]]),
                        Matrix([[-1.31386475,  0.88462238,  0.88131804,  1.70957306],
                               [ 0.05003364, -0.40467741, -0.54535995, -1.54647732],
                               [ 0.98236743, -1.10106763, -1.18504653, -0.2056499 ]]),
                        Matrix([[ 1.48614836],
                               [ 0.23671627],
                               [-1.02378514]])),
                       Matrix([[-0.7129932 ,  0.62524497],
                               [-0.16051336, -0.76883635],
                               [-0.23003072,  0.74505627]])),
                      ((Matrix([[ 1.97611078, -1.24412333],
                               [-0.62641691, -0.80376609],
                               [-2.41908317, -0.92379202]]),
                        Matrix([[-1.02387576,  1.12397796, -0.13191423]]),
                        Matrix([[-1.62328545]])),
                        Matrix([[ 0.64667545, -0.35627076]]))]
        let dnn = DeepNeuralNetwork(layerDimensions: [4,5,1], X: X, Y:Y)
        let (dA, dW, db) = dnn.L_model_backward(AL, Y_assess, caches)
        
        XCTAssertEqual(dA[1]![0,0], 0.12913162, accuracy: 1e-7 )
        XCTAssertEqual(dA[1]![0,1], -0.44014127, accuracy: 1e-7 )
        XCTAssertEqual(dW[1]![0,0], 0.41010002, accuracy: 1e-7 )
        XCTAssertEqual(dW[1]![0,1], 0.07807203, accuracy: 1e-7 )
        XCTAssertEqual(db[1]![0,0], -0.22007063, accuracy: 1e-7 )
        XCTAssertEqual(db[1]![2,0], -0.02835349, accuracy: 1e-7 )
    }
    
    func testUpdateParameters() {
        // --  Test update_parameters
        // Expected results
         
         //W1    [[-0.59562069 -0.09991781 -2.14584584 1.82662008] [-1.76569676 -0.80627147 0.51115557 -1.18258802] [-1.0535704 -0.86128581 0.68284052 2.20374577]]
         //b1    [[-0.04659241] [-1.28888275] [ 0.53405496]]
         //W2    [[-0.55569196 0.0354055 1.32964895]]
         //b2    [[-0.84610769]]

        var W : [Int:Matrix] = [:]
        var b : [Int:Matrix] = [:]
        var dW : [Int:Matrix] = [:]
        var db : [Int:Matrix] = [:]

        W[1]=Matrix([[-0.41675785, -0.05626683, -2.1361961 ,  1.64027081],
               [-1.79343559, -0.84174737,  0.50288142, -1.24528809],
               [-1.05795222, -0.90900761,  0.55145404,  2.29220801]])
        b[1]=Matrix([[ 0.04153939],
               [-1.11792545],
               [ 0.53905832]])
        W[2]=Matrix([[-0.5961597 , -0.0191305 ,  1.17500122]])
        b[2]=Matrix([[-0.74787095]])
        dW[1]=Matrix([[ 1.78862847,  0.43650985,  0.09649747, -1.8634927 ],
               [-0.2773882 , -0.35475898, -0.08274148, -0.62700068],
               [-0.04381817, -0.47721803, -1.31386475,  0.88462238]])
        db[1]=Matrix([[ 0.88131804],
               [ 1.70957306],
               [ 0.05003364]])
        dW[2]=Matrix([[-0.40467741, -0.54535995, -1.54647732]])
        db[2]=Matrix([[ 0.98236743]])

        let X = Matrix([[-0.31178367,  0.72900392,  0.21782079, -0.8990918 ],
                     [-2.48678065, 0.91325152,  1.12706373, -1.51409323],
                     [ 1.63929108, -0.4298936,   2.63128056,  0.60182225],
                     [-0.33588161,  1.23773784,  0.11112817,  0.12915125],
                     [ 0.07612761, -0.15512816,  0.63422534,  0.810655  ]])
        let Y = Matrix([[1, 1, 0]])
        let dnn = DeepNeuralNetwork(layerDimensions: [3,4,5], X: X, Y:Y)
        (W, b) = dnn.update_parameters((W,b), (dW, dW, db), 0.1)
        
        XCTAssertEqual(W[1]![0,0], -0.595620697, accuracy: 1e-7 )
        XCTAssertEqual(W[1]![0,1], -0.099917815, accuracy: 1e-7 )
        XCTAssertEqual(W[2]![0,1], 0.035405495, accuracy: 1e-7 )
        XCTAssertEqual(b[1]![0,0], -0.046592414000000006, accuracy: 1e-7 )
        XCTAssertEqual(b[1]![1,0], -1.288882756, accuracy: 1e-7 )
        XCTAssertEqual(b[2]![0,0], -0.846107693, accuracy: 1e-7 )
    }
    
    func testRun2LayerModel() {
        let X = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/EX1_train_x", withExtension: "csv")!.path)
        XCTAssertEqual( X.rows, 12288, "2-Layer-Model X shape test failed" )
        XCTAssertEqual( X.columns, 209, "2-Layer-Model X shape test failed" )

        let Y = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/EX1_train_y", withExtension: "csv")!.path)
        XCTAssertEqual( Y.rows, 1, "2-Layer-Model Y shape test failed" )
        XCTAssertEqual( Y.columns, 209, "2-Layer-Model Y shape test failed" )

        var W1 : Matrix = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/2L-W1", withExtension: "csv")!.path)
        W1 = W1′
        XCTAssertEqual( W1.rows, 7, "2-Layer-Model W1 shape test failed" )
        XCTAssertEqual( W1.columns, 12288, "2-Layer-Model W1 shape test failed" )

        let W2 : Matrix = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/2L-W2", withExtension: "csv")!.path)
        XCTAssertEqual( W2.rows, 1, "2-Layer-Model W2 shape test failed" )
        XCTAssertEqual( W2.columns, 7, "2-Layer-Model W2 shape test failed" )

        let X_test : Matrix = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/EX1_test_x", withExtension: "csv")!.path)

        let Y_test : Matrix = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/EX1_test_y", withExtension: "csv")!.path)

        let dnn = DeepNeuralNetwork(layerDimensions: [12288, 7, 1], X: X, Y: Y)
        dnn.num_iterations = 2500
        dnn.learning_rate = 0.0075
        let( accuracy, costs ) = dnn.two_layer_model_train(W1, W2)
        
        XCTAssertEqual(costs[2400]!, 0.048554785628770136, accuracy: 1e-7, "2-Layer-Model cost at 2400 test failed" )
        XCTAssertEqual(accuracy, 1.0, accuracy: 1e-7, "2-Layer-Model accuracy test failed" )

        let testAccuracy = dnn.two_layer_model_predict(X_test, Y_test)
        XCTAssertEqual(testAccuracy, 0.72, accuracy: 1e-7, "2-Layer-Model accuracy on test set test failed" )
 
    }
    
    func testRunLLayerModel() {
        let X = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/EX1_train_x", withExtension: "csv")!.path)
        XCTAssertEqual( X.rows, 12288, "2-Layer-Model X shape test failed" )
        XCTAssertEqual( X.columns, 209, "2-Layer-Model X shape test failed" )

        let Y = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/EX1_train_y", withExtension: "csv")!.path)
        XCTAssertEqual( Y.rows, 1, "2-Layer-Model Y shape test failed" )
        XCTAssertEqual( Y.columns, 209, "2-Layer-Model Y shape test failed" )

        let X_test : Matrix = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/EX1_test_x", withExtension: "csv")!.path)

        let Y_test : Matrix = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/EX1_test_y", withExtension: "csv")!.path)

        var W1 : Matrix = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/LL-W1", withExtension: "csv")!.path)
        W1 = W1′

        let W2 : Matrix = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/LL-W2", withExtension: "csv")!.path)

        let W3 : Matrix = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/LL-W3", withExtension: "csv")!.path)

        let W4 : Matrix = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/LL-W4", withExtension: "csv")!.path)

        let dnn = DeepNeuralNetwork(layerDimensions: [12288, 20, 7, 5, 1], X: X, Y: Y)
        dnn.learning_rate = 0.0075
        dnn.num_iterations = 2500
        dnn.optimization_type = .gradientdescent
        
        let (accuracy, costs) = dnn.L_layer_model_train([1:W1, 2:W2, 3:W3, 4:W4])
        
        XCTAssertEqual(costs[2400]!, 0.07178487334418725, accuracy: 1e-7, "L-Layer-Model cost at 2400 test failed" )
        XCTAssertEqual(accuracy, 0.9904306220095693, accuracy: 1e-7, "L-Layer-Model accuracy test failed" )

        let testAccuracy = dnn.L_layer_model_predict(X_test, Y_test)
        XCTAssertEqual(testAccuracy, 0.78, accuracy: 1e-7, "L-Layer-Model accuracy on test set test failed" )
    }
    func testXORwithWeightsMatrices() {
        let X = Matrix([
            [0,0], [0,1], [1,0], [1,1]
        ])′
        let Y = Matrix([[0,1,1,0]])
        let W1 = Matrix([[0.003488999386915159, 0.3458335032337748], [0.8704062974282158, 0.8258641591950425]])
        let W2 = Matrix([[0.9727850364674658, 0.5037613727131417]])
        let dnn = DeepNeuralNetwork(layerDimensions: [2, 2, 1], X: X, Y: Y)
        dnn.learning_rate = 0.15
        dnn.num_iterations = 1000
        dnn.batch_type = .stochastic
        let (accuracy, _) = dnn.L_layer_model_train([1:W1, 2:W2])
        XCTAssertEqual(accuracy, 1.0, accuracy: 1e-7, "XOR accuracy test failed" )
    }

    func testComputeCostRegularized() {
        var W : [Int: Matrix] = [:]
        
        W[1] = Matrix([[ 1.62434536,-0.61175641, -0.52817175],
                        [-1.07296862,  0.86540763, -2.3015387 ]])
        W[2] = Matrix([[ 0.3190391 , -0.24937038],
                         [ 1.46210794, -2.06014071],
                         [-0.3224172,  -0.38405435]])
        W[3] = Matrix([[-0.87785842,  0.04221375,  0.58281521]])
        let λ = 0.1
        let m = 5.0
        var L2_regularization_cost : Double = 0
        if( λ>0.0 ) {
            // Compute L2 Regularization
            let L = 4
            for l in 1..<L {
                L2_regularization_cost += (Σ(W[l]!*W[l]!, .both))[0,0]
            }
            L2_regularization_cost *= (λ / (2 * m))
        }
        XCTAssertEqual(L2_regularization_cost, 0.18398434041223857, accuracy: 1e-7 )
    }
    
    func testRegularization() {
        let X = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/DLS2_W1_trainX", withExtension: "csv")!.path)
        XCTAssertEqual( X.rows, 2 )
        XCTAssertEqual( X.columns, 211 )

        let Y = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/DLS2_W1_trainY", withExtension: "csv")!.path)
        XCTAssertEqual( Y.rows, 1 )
        XCTAssertEqual( Y.columns, 211 )

        let X_test : Matrix = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/DLS2_W1_testX", withExtension: "csv")!.path)
        XCTAssertEqual( X_test.rows, 2 )
        XCTAssertEqual( X_test.columns, 200 )

        let Y_test : Matrix = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/DLS2_W1_testY", withExtension: "csv")!.path)
        XCTAssertEqual( Y_test.rows, 1 )
        XCTAssertEqual( Y_test.columns, 200 )
        let W1 = Matrix([[ 1.26475132,  0.30865908],
                         [ 0.06823401, -1.31768833],
                         [-0.19614308, -0.25085248],
                         [-0.05850706, -0.44335643],
                         [-0.03098412, -0.33744411],
                         [-0.92904268,  0.62552248],
                         [ 0.62318596,  1.20885071],
                         [ 0.03537913, -0.28615014],
                         [-0.38562772, -1.0935246 ],
                         [ 0.69463867, -0.77857239],
                         [-0.83795444, -0.14541644],
                         [ 1.05086558,  0.16738368],
                         [-0.72392541, -0.50416233],
                         [ 0.44211496, -0.11350009],
                         [-0.5436494 , -0.16265628],
                         [ 0.52683434,  1.39732134],
                         [-0.87972804, -0.44294365],
                         [-0.56834846, -1.71055012],
                         [-0.6532196 , -0.72398949],
                         [ 0.79477244, -0.09327745]])
        let W2 = Matrix([[-0.36297766,  0.14460103, -0.07966456, -0.38977819, -0.13341492,
                          -0.1316137 , -0.19540602,  0.00664421, -0.50272572, -0.05987337,
                           0.2265547 ,  0.19069139,  0.24779826,  0.25030336,  0.33262476,
                          -0.25005963,  0.1891341 , -0.41610755, -0.13480921, -0.42808896],
                         [ 0.23437291,  0.29823284, -0.04414326,  0.39682269, -0.15087366,
                           0.03367895,  0.0341997 , -0.2379613 ,  0.09792784,  0.43356876,
                          -0.22918151,  0.20109819, -0.03454878,  0.39570069,  0.10817836,
                           0.15120658,  0.14381568,  0.05569748, -0.31210221,  0.31118529],
                         [-0.30649091,  0.05334435,  0.13731181, -0.18736288,  0.03243712,
                           0.26114642, -0.00538992, -0.19870984, -0.65197878, -0.21731014,
                          -0.13216922, -0.11547443, -0.21466167,  0.08436578, -0.12850871,
                          -0.02447473,  0.15184503, -0.19128157, -0.06712812,  0.48257686]])
        let W3 = Matrix([[ 0.5047691 , -0.74682372, -0.04603845]])
        
        let dnn = DeepNeuralNetwork(layerDimensions: [2, 20, 3, 1], X: X, Y: Y)
        dnn.weigth_init_type = .he
        dnn.learning_rate = 0.3
        dnn.num_iterations = 30000
        dnn.λ = 0.7
        let (accuracy, _) = dnn.L_layer_model_train([1:W1, 2:W2, 3:W3])
        XCTAssertGreaterThan(accuracy, 0.91)
        let testAccuracy = dnn.L_layer_model_predict(X_test, Y_test)
        XCTAssertGreaterThanOrEqual(testAccuracy, 0.93)
    }
    func testDropout() {
        let X = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/DLS2_W1_trainX", withExtension: "csv")!.path)
        XCTAssertEqual( X.rows, 2 )
        XCTAssertEqual( X.columns, 211 )

        let Y = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/DLS2_W1_trainY", withExtension: "csv")!.path)
        XCTAssertEqual( Y.rows, 1 )
        XCTAssertEqual( Y.columns, 211 )

        let X_test : Matrix = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/DLS2_W1_testX", withExtension: "csv")!.path)
        XCTAssertEqual( X_test.rows, 2 )
        XCTAssertEqual( X_test.columns, 200 )

        let Y_test : Matrix = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/DLS2_W1_testY", withExtension: "csv")!.path)
        XCTAssertEqual( Y_test.rows, 1 )
        XCTAssertEqual( Y_test.columns, 200 )
        let W1 = Matrix([[ 1.26475132,  0.30865908],
                         [ 0.06823401, -1.31768833],
                         [-0.19614308, -0.25085248],
                         [-0.05850706, -0.44335643],
                         [-0.03098412, -0.33744411],
                         [-0.92904268,  0.62552248],
                         [ 0.62318596,  1.20885071],
                         [ 0.03537913, -0.28615014],
                         [-0.38562772, -1.0935246 ],
                         [ 0.69463867, -0.77857239],
                         [-0.83795444, -0.14541644],
                         [ 1.05086558,  0.16738368],
                         [-0.72392541, -0.50416233],
                         [ 0.44211496, -0.11350009],
                         [-0.5436494 , -0.16265628],
                         [ 0.52683434,  1.39732134],
                         [-0.87972804, -0.44294365],
                         [-0.56834846, -1.71055012],
                         [-0.6532196 , -0.72398949],
                         [ 0.79477244, -0.09327745]])
        let W2 = Matrix([[-0.36297766,  0.14460103, -0.07966456, -0.38977819, -0.13341492,
                          -0.1316137 , -0.19540602,  0.00664421, -0.50272572, -0.05987337,
                           0.2265547 ,  0.19069139,  0.24779826,  0.25030336,  0.33262476,
                          -0.25005963,  0.1891341 , -0.41610755, -0.13480921, -0.42808896],
                         [ 0.23437291,  0.29823284, -0.04414326,  0.39682269, -0.15087366,
                           0.03367895,  0.0341997 , -0.2379613 ,  0.09792784,  0.43356876,
                          -0.22918151,  0.20109819, -0.03454878,  0.39570069,  0.10817836,
                           0.15120658,  0.14381568,  0.05569748, -0.31210221,  0.31118529],
                         [-0.30649091,  0.05334435,  0.13731181, -0.18736288,  0.03243712,
                           0.26114642, -0.00538992, -0.19870984, -0.65197878, -0.21731014,
                          -0.13216922, -0.11547443, -0.21466167,  0.08436578, -0.12850871,
                          -0.02447473,  0.15184503, -0.19128157, -0.06712812,  0.48257686]])
        let W3 = Matrix([[ 0.5047691 , -0.74682372, -0.04603845]])
        
        let dnn = DeepNeuralNetwork(layerDimensions: [2, 20, 3, 1], X: X, Y: Y)
        dnn.learning_rate = 0.3
        dnn.num_iterations = 30000
        dnn.keep_prob = 0.86
        let (accuracy, _) = dnn.L_layer_model_train([1:W1, 2:W2, 3:W3])
        XCTAssertGreaterThan(accuracy, 0.92)
        let testAccuracy = dnn.L_layer_model_predict(X_test, Y_test)
        XCTAssertGreaterThanOrEqual(testAccuracy, 0.95)
    }
    func testAdamRoutine() {
        let β1 = 0.9
        let β2 = 0.999
        let t = 2.0
        let learning_rate = 0.01
        let L = 2
        let epsilon = 1e-8
        
        var vdW : [ Int : Matrix ] = [:]
        var vdb : [ Int : Matrix ] = [:]
        var sdW : [ Int : Matrix ] = [:]
        var sdb : [ Int : Matrix ] = [:]
        var W_tmp : [ Int : Matrix ] = [:]
        var b_tmp : [ Int : Matrix ] = [:]
        var dW : [ Int : Matrix ] = [:]
        var db : [ Int : Matrix ] = [:]
        var vdW_corrected : [ Int : Matrix ] = [:]
        var vdb_corrected : [ Int : Matrix ] = [:]
        var sdW_corrected : [ Int : Matrix ] = [:]
        var sdb_corrected : [ Int : Matrix ] = [:]

        W_tmp[1] = Matrix([[ 1.62434536, -0.61175641, -0.52817175],
                       [-1.07296862,  0.86540763, -2.3015387 ]])
        W_tmp[2] = Matrix([[ 0.3190391 , -0.24937038,  1.46210794],
                           [-2.06014071, -0.3224172 , -0.38405435],
                           [ 1.13376944, -1.09989127, -0.17242821]])
        b_tmp[1] = Matrix([[ 1.74481176],
                           [-0.7612069 ]])
        b_tmp[2] = Matrix([[-0.87785842],
                           [ 0.04221375],
                           [ 0.58281521]])
        dW[1] = Matrix([[-1.10061918,  1.14472371,  0.90159072],
                        [ 0.50249434,  0.90085595, -0.68372786]])
        dW[2] = Matrix([[-0.26788808,  0.53035547, -0.69166075],
                        [-0.39675353, -0.6871727 , -0.84520564],
                        [-0.67124613, -0.0126646 , -1.11731035]])
        db[1] = Matrix([[-0.12289023],
                        [-0.93576943]])
        db[2] = Matrix([[ 0.2344157 ],
                        [ 1.65980218],
                        [ 0.74204416]])
        vdW[1] = Matrix([[ 0.0,  0.0,  0.0],
                         [ 0.0,  0.0,  0.0]])
        
        vdW[2] = Matrix([[ 0.0,  0.0,  0.0],
                         [ 0.0,  0.0,  0.0],
                         [ 0.0,  0.0,  0.0]])
        vdb[1] = Matrix([[ 0.0],
                         [ 0.0]])
        vdb[2] = Matrix([[ 0.0],
                            [ 0.0],
                            [ 0.0]])
        sdW[1] = Matrix([[ 0.0,  0.0,  0.0],
                         [ 0.0,  0.0,  0.0]])
        
        sdW[2] = Matrix([[ 0.0,  0.0,  0.0],
                         [ 0.0,  0.0,  0.0],
                         [ 0.0,  0.0,  0.0]])
        sdb[1] = Matrix([[ 0.0],
                         [ 0.0]])
        sdb[2] = Matrix([[ 0.0],
                            [ 0.0],
                            [ 0.0]])
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
        XCTAssertEqual(W_tmp[1]![0,0], 1.63178673, accuracy: 1e-7 )
        XCTAssertEqual(W_tmp[1]![1,1], 0.85796626, accuracy: 1e-7 )
        XCTAssertEqual(b_tmp[1]![0,0], 1.75225313, accuracy: 1e-7 )
        XCTAssertEqual(b_tmp[1]![1,0], -0.75376553, accuracy: 1e-7 )
        XCTAssertEqual(W_tmp[2]![0,0], 0.32648046, accuracy: 1e-7 )
        XCTAssertEqual(W_tmp[2]![1,1], -0.31497584, accuracy: 1e-7 )
        XCTAssertEqual(b_tmp[2]![0,0], -0.88529979, accuracy: 1e-7 )
        XCTAssertEqual(b_tmp[2]![1,0], 0.03477238, accuracy: 1e-7 )
        XCTAssertEqual(vdW[1]![0,0], -0.11006192, accuracy: 1e-7 )
        XCTAssertEqual(vdW[1]![1,1], 0.09008559, accuracy: 1e-7 )
        XCTAssertEqual(vdb[1]![0,0], -0.01228902, accuracy: 1e-7 )
        XCTAssertEqual(vdb[1]![1,0], -0.09357694, accuracy: 1e-7 )
        XCTAssertEqual(vdW[2]![0,0], -0.02678881, accuracy: 1e-7 )
        XCTAssertEqual(vdW[2]![1,1], -0.06871727, accuracy: 1e-7 )
        XCTAssertEqual(vdb[2]![0,0], 0.02344157, accuracy: 1e-7 )
        XCTAssertEqual(vdb[2]![1,0], 0.16598022, accuracy: 1e-7 )
        XCTAssertEqual(sdW[1]![0,0], 0.00121136, accuracy: 1e-7 )
        XCTAssertEqual(sdW[1]![1,1], 0.00081154, accuracy: 1e-7 )
        XCTAssertEqual(sdb[1]![0,0], 1.51020075e-05, accuracy: 1e-7 )
        XCTAssertEqual(sdb[1]![1,0], 8.75664434e-04, accuracy: 1e-7 )
        XCTAssertEqual(sdW[2]![0,0], 7.17640232e-05 , accuracy: 1e-7 )
        XCTAssertEqual(sdW[2]![1,1], 4.72206320e-04, accuracy: 1e-7 )
        XCTAssertEqual(sdb[2]![0,0], 5.49507194e-05, accuracy: 1e-7 )
        XCTAssertEqual(sdb[2]![1,0], 2.75494327e-03, accuracy: 1e-7 )
    }
    func testRandomMiniBatches() {
        let X = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/EX1_train_x", withExtension: "csv")!.path)

        let Y = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/EX1_train_y", withExtension: "csv")!.path)

        let dnn = DeepNeuralNetwork(layerDimensions: [12288, 20, 7, 5, 1], X: X, Y: Y)
        dnn.learning_rate = 0.0075
        dnn.num_iterations = 2500
        let mini_batches : [(Matrix, Matrix)] = dnn.random_mini_batches()
        XCTAssertEqual( mini_batches.count, 4 )
        let (batchX, _) = mini_batches[3]
        XCTAssertEqual( batchX.columns, 17 )
    }
    
    func testMomentumOptimization() {
        let X = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/DLS2_W2_trainX", withExtension: "csv")!.path)
        XCTAssertEqual( X.rows, 2 )
        XCTAssertEqual( X.columns, 300 )

        let Y = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/DLS2_W2_trainY", withExtension: "csv")!.path)
        XCTAssertEqual( Y.rows, 1 )
        XCTAssertEqual( Y.columns, 300 )
        let dnn = DeepNeuralNetwork(layerDimensions: [2, 5, 2, 1], X: X, Y: Y)
        dnn.learning_rate = 0.0007
        dnn.num_iterations = 10000
        dnn.mini_batch_size = 64
        dnn.batch_type = .minibatch
        dnn.optimization_type = .momentum
        dnn.weigth_init_type = .he
        let (accuracy, costs) = dnn.L_layer_model_train(nil)
        XCTAssertLessThan(costs[9900]!, 0.50)
        XCTAssertGreaterThan(accuracy, 0.79)
    }

    func testAdamOptimization() {
        let X = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/DLS2_W2_trainX", withExtension: "csv")!.path)
        XCTAssertEqual( X.rows, 2 )
        XCTAssertEqual( X.columns, 300 )

        let Y = DeepNeuralNetwork.csv(fileUrl: Bundle.module.url(forResource: "TestData/DLS2_W2_trainY", withExtension: "csv")!.path)
        XCTAssertEqual( Y.rows, 1 )
        XCTAssertEqual( Y.columns, 300 )
        let dnn = DeepNeuralNetwork(layerDimensions: [2, 5, 2, 1], X: X, Y: Y)
        dnn.learning_rate = 0.0007
        dnn.num_iterations = 10000
        dnn.mini_batch_size = 64
        dnn.batch_type = .minibatch
        dnn.optimization_type = .adam
        dnn.weigth_init_type = .he
        let (accuracy, costs) = dnn.L_layer_model_train(nil)
        XCTAssertLessThan(costs[9900]!, 0.14)
        XCTAssertGreaterThan(accuracy, 0.94)
    }
    func testMulticlassAccuracy() {
        let Y = Matrix([[0,1,0,0,0,0,0,1],
                        [1,0,0,0,1,0,0,0],
                        [0,0,1,0,0,0,0,0],
                        [0,0,0,1,0,1,1,0]])
        let Ypred = Matrix([[0.3, 0.1, 0.2, 0.0, 0.1, 0.2, 0.2, 0.8],
                            [0.4, 0.3, 0.1, 0.8, 0.2, 0.0, 0.2, 0.1],
                            [0.2, 0.4, 0.4, 0.1, 0.3, 0.1, 0.2, 0.1],
                            [0.1, 0.2, 0.3, 0.1, 0.4, 0.7, 0.4, 0.0]])+1e-8
        
        var correct = 0.0
        for j in 0..<Y.columns {
            if (Ypred[.all, j]′ ° Y[.all,j])[0,0] == max(Ypred[.all, j]) { correct += 1.0 }
        }
        let accuracy = correct / Double(Ypred.columns)
        XCTAssertEqual(accuracy, 0.625)
    }

    func testMultitaskCost() {
        let Y = Matrix([[0,1,0,0,1,0,-1,1],
                        [1,0,0,0,1,0,0,0],
                        [1,-1,1,0,0,0,0,0],
                        [0,0,0,1,0,1,1,0]])
        let Yhat = Matrix([[0.3, 0.1, 0.2, 0.0, 0.1, 0.2, 0.2, 0.8],
                            [0.4, 0.3, 0.1, 0.8, 0.2, 0.0, 0.2, 0.1],
                            [0.2, 0.4, 0.4, 0.1, 0.3, 0.1, 0.2, 0.1],
                            [0.1, 0.2, 0.3, 0.1, 0.4, 0.7, 0.4, 0.0]])+1e-8
        let filteredY = Y >= 0.0
        let m = Y.columns
        let cost = (Σ(-Σ( filteredY*(Y*log(Yhat) + (1-Y) * log(1-Yhat)),.column), .row)/m)[0,0]
        XCTAssertEqual(cost, 2.343770717298659)
    }
    static var allTests = [
        ("testLinearForward", testLinearForward),
        ("testLinearActivationForward", testLinearActivationForward),
        ("testLModelForward", testLModelForward),
        ("testCostFunction", testCostFunction),
        ("testLinearBackward", testLinearBackward),
        ("testLinearActivationBackward", testLinearActivationBackward),
        ("testLModelBackward", testLModelBackward),
        ("testUpdateParameters", testUpdateParameters),
        ("testRun2LayerModel", testRun2LayerModel),
        ("testRunLLayerModel", testRunLLayerModel),
        ("testXORwithWeightsMatrices", testXORwithWeightsMatrices),
        ("testComputeCostRegularized", testComputeCostRegularized),
        ("testRegularization", testRegularization),
        ("testDropout", testDropout),
        ("testRandomMiniBatches", testRandomMiniBatches),
        ("testMomentumOptimization", testMomentumOptimization),
        ("testAdamOptimization", testAdamOptimization),
        ("testMulticlassAccuracy", testMulticlassAccuracy),
    ]
}
