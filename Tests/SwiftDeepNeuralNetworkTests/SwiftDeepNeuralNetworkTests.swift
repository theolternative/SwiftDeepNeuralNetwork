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
        let b = Matrix([[-0.24937038, -0.24937038]])

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
        let b =  Matrix([[-0.90900761, -0.90900761]])

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
        dnn.b[1] = Matrix([[ 1.38503523, 1.38503523, 1.38503523, 1.38503523],
                       [-0.51962709, -0.51962709, -0.51962709, -0.51962709],
                       [-0.78015214, -0.78015214, -0.78015214, -0.78015214],
                       [ 0.95560959, 0.95560959, 0.95560959, 0.95560959]])
        dnn.W[2] = Matrix([[-0.12673638, -1.36861282,  1.21848065, -0.85750144],
                       [-0.56147088, -1.0335199 ,  0.35877096,  1.07368134],
                       [-0.37550472,  0.39636757, -0.47144628,  2.33660781]])
        dnn.b[2] = Matrix([[ 1.50278553, 1.50278553, 1.50278553, 1.50278553],
                       [-0.59545972, -0.59545972, -0.59545972, -0.59545972],
                       [ 0.52834106, 0.52834106, 0.52834106, 0.52834106]])
        dnn.W[3] = Matrix( [[ 0.9398248 ,  0.42628539, -0.75815703]] )
        dnn.b[3] = Matrix( [[-0.16236698, -0.16236698, -0.16236698, -0.16236698]])
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
        let cost = dnn.compute_cost(AL)
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
        let (accuracy, _) = dnn.L_layer_model_train([1:W1, 2:W2])
        XCTAssertEqual(accuracy, 1.0, accuracy: 1e-7, "XOR accuracy test failed" )
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

        let dnn = DeepNeuralNetwork(layerDimensions: [2, 20, 3, 1], X: X, Y: Y)
        dnn.weigth_init_type = .random
        dnn.custom_weight_factor = 2
        dnn.learning_rate = 0.3
        dnn.num_iterations = 30000
        let (accuracy, _) = dnn.L_layer_model_train(nil)
        XCTAssertEqual(accuracy, 1.0, accuracy: 1e-7, "XOR accuracy test failed" )
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
        ("testRegularization", testRegularization),
    ]
}
