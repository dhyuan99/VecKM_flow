<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Learning Normal Flow Directly from Event Neighborhoods </h1>

<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://www.cs.umd.edu/~dhyuan" target="_blank" style="text-decoration: none;">Dehao Yuan</a>&nbsp;,&nbsp;
    <a href="http://users.umiacs.umd.edu/~yiannis/" target="_blank" style="text-decoration: none;">Levi Burner</a>&nbsp;&nbsp;
    <a href="http://users.umiacs.umd.edu/~yiannis/" target="_blank" style="text-decoration: none;">Jiayi Wu</a>&nbsp;&nbsp;
    <a href="http://users.umiacs.umd.edu/~yiannis/" target="_blank" style="text-decoration: none;">Minghui Liu</a>&nbsp;&nbsp;
    <a href="http://users.umiacs.umd.edu/~yiannis/" target="_blank" style="text-decoration: none;">Jingxi Chen</a>&nbsp;&nbsp;
    <a href="http://users.umiacs.umd.edu/~yiannis/" target="_blank" style="text-decoration: none;">Yiannis Aloimonos</a>&nbsp;&nbsp;
    <a href="http://users.umiacs.umd.edu/~fer/" target="_blank" style="text-decoration: none;">Cornelia Fermüller</a>
</p>

<p align='center';>
<b>
<em>CVPR2025</em> &nbsp&nbsp&nbsp&nbsp <a href="http://arxiv.org/abs/2404.01568" target="_blank" style="text-decoration: none;">[Paper]</a>
</b>
</p>

## Highlighted Features
It is a generic normal flow estimator with event camera inputs. The API is called by following, after installing the package.
```
estimator = VecKMNormalFlowEstimator(training_set='UNION')
flow_predictions, flow_uncertainty = estimator.inference(events_t, events_xy)
flow_predictions[flow_uncertainty > 0.3] = np.nan
```