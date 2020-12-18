pipeline {
    agent any

    // triggers {
    //     pollSCM('*/5 * * * 1-5')
    // }

    // options {
    //     skipDefaultCheckout(true)
    //     // Keep the 10 most recent builds
    //     buildDiscarder(logRotator(numToKeepStr: '10'))
    //     timestamps()
    // }

    environment {
      imagename = "valerielimyh/ml_deploy_aws"
      registryCredential = 'valerie-dockerhub'
      PATH="/var/lib/jenkins/miniconda3/bin:$PATH"
    }

    // environment {
    //   PATH="$WORKSPACE/miniconda3/bin:$PATH"
    // }    

    stages {
        // def dockerImage
        stage ("Code pull"){
            steps{
                git([url: 'https://github.com/valerielimyh/ml_deploy_aws', branch: 'master', credentialsId: 'valerie-github-user-token'])
                // checkout scm
            }
        }

        // stage('setup miniconda') {
        //     steps {
        //     sh '''#!/usr/bin/env bash
        //     wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        //     bash miniconda.sh -b -p $WORKSPACE/miniconda
        //     hash -r
        //     conda config --set always_yes yes --set changeps1 no
        //     conda update -q conda
        //     chmod -R 777 $WORKSPACE/miniconda

        //     # Useful for debugging any issues with conda
        //     conda info -a
        //     conda config --add channels defaults
        //     conda config --add channels conda-forge
        //     conda config --add channels bioconda

        //     # create torchVenv env
        //     conda init bash
        //     conda env create -f envs/torchVenv.yaml
        //     '''
        //     }
        // }

        // stage('Build environment') {
        //     steps {
        //         echo "Building virtualenv"
        //         sh  ''' #!/usr/bin/env bash
        //                 conda init bash
        //                 conda activate miniconda/envs/torchVenv/
        //                 pip install --no-cache-dir -r requirements.txt
        //             '''
        //     }
        // }
        stage('Build environment') {
            steps {
                sh '''conda create --yes -n ${BUILD_TAG} python=3.7
                      source activate ${BUILD_TAG} 
                      pip install --no-cache-dir -r requirements.txt
                    '''
            }
        }
        // stage('Static code metrics') {
        //     steps {
        //         echo "Raw metrics"
        //         sh  ''' source activate ${BUILD_TAG}
        //                 radon raw --json irisvmpy > raw_report.json
        //                 radon cc --json irisvmpy > cc_report.json
        //                 radon mi --json irisvmpy > mi_report.json
        //                 sloccount --duplicates --wide irisvmpy > sloccount.sc
        //             '''
        //         echo "Test coverage"
        //         sh  ''' source activate ${BUILD_TAG}
        //                 coverage run irisvmpy/iris.py 1 1 2 3
        //                 python -m coverage xml -o reports/coverage.xml
        //             '''
        //         echo "Style check"
        //         sh  ''' source activate ${BUILD_TAG}
        //                 pylint irisvmpy || true
        //             '''
        //     }
        //     post{
        //         always{
        //             step([$class: 'CoberturaPublisher',
        //                            autoUpdateHealth: false,
        //                            autoUpdateStability: false,
        //                            coberturaReportFile: 'reports/coverage.xml',
        //                            failNoReports: false,
        //                            failUnhealthy: false,
        //                            failUnstable: false,
        //                            maxNumberOfBuilds: 10,
        //                            onlyStable: false,
        //                            sourceEncoding: 'ASCII',
        //                            zoomCoverageChart: false])
        //         }
        //     }
        // }


        // stage('Unit tests') {
        //     steps {
        //         sh  ''' conda activate miniconda/envs/torchVenv/
        //                 pytest --v --junit-xml tests/reports/unit_tests.xml
        //             '''
        //     }
        //     post {
        //         always {
        //             // Archive unit tests for the future
        //             junit (allowEmptyResults: true,
        //                   testResults: './tests/reports/unit_tests.xml',
        //                 //   fingerprint: true
        //                   )
        //         }
        //     }
        // }

        stage('Unit tests') {
            steps {
                sh  ''' source activate ${BUILD_TAG}
                        pytest -v --junit-xml tests/reports/unit_tests.xml
                    '''
            }
            post {
                always {
                    // Archive unit tests for the future
                    junit (allowEmptyResults: true,
                          testResults: './tests/reports/unit_tests.xml',
                        //   fingerprint: true
                          )
                }
                success {

                    echo "Success"
                }

                failure {

                    echo 'Failure. Something went wrong with the build. Printing environment for debugging'            
                    sh '''printenv'''
                    echo 'Searching for test directories/files in the system...'
                    sh '''find / -name "test*"'''
                }

            }
        }
        stage('Building image') {

            steps {
             script {
            // Build the Docker image
            dockerImage = docker.build imagename
                }
            }
            }

        stage('Deploy Image') {
            steps{
            script {
            docker.withRegistry( '', registryCredential ) {
            dockerImage.push("$BUILD_NUMBER")
            dockerImage.push('latest')
                    }
                }
            }
        }

        stage('Remove Unused docker image') {
            steps{
            sh '''docker rmi $imagename:$BUILD_NUMBER'''
            sh '''docker rmi $imagename:latest'''
                }
            }
    }
        // stage('Python pytest Tests') {
        //     dir('python/pytest') {
        //         sh 'virtualenv -p /usr/bin/python3 venv'
        //         sh 'source venv/bin/activate && pip install -r requirements.txt'
        //         sh 'source venv/bin/activate && pytest --junit-xml=test_results.xml test || true'
        //         junit keepLongStdio: true, allowEmptyResults: true, testResults: 'test_results.xml'
        //     }
        // }

        // stage('Integration tests') {
        //     steps {
        //         sh  ''' source activate ${BUILD_TAG}
        //                 behave -f=formatters.cucumber_json:PrettyCucumberJSONFormatter -o ./reports/integration.json
        //             '''
        //     }
        //     post {
        //         always {
        //             cucumber (fileIncludePattern: '**/*.json',
        //                       jsonReportDirectory: './reports/',
        //                       parallelTesting: true,
        //                       sortingMethod: 'ALPHABETICAL')
        //         }
        //     }
        // }

        // stage('Build package') {
        //     when {
        //         expression {
        //             currentBuild.result == null || currentBuild.result == 'SUCCESS'
        //         }
        //     }
        //     steps {
        //         sh  ''' source activate ${BUILD_TAG}
        //                 python setup.py bdist_wheel  
        //             '''
        //     }
        //     post {
        //         always {
        //             // Archive unit tests for the future
        //             archiveArtifacts (allowEmptyArchive: true,
        //                              artifacts: 'dist/*whl',
        //                              fingerprint: true)
        //         }
        //     }
        // }

        // stage("Deploy to PyPI") {
        //     }
        //     steps {
        //         sh "twine upload dist/*"
        //     }
        

    post {
        always {
            sh '''conda remove --yes -n ${BUILD_TAG} --all'''
        }
        failure {
            emailext (
                subject: "FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'",
                body: """<p>FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]':</p>
                         <p>Check console output at &QUOT;<a href='${env.BUILD_URL}'>${env.JOB_NAME} [${env.BUILD_NUMBER}]</a>&QUOT;</p>""",
                recipientProviders: [[$class: 'DevelopersRecipientProvider']]
            )
        }
    }
}