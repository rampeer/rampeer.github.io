### Nodes, applications, clusters and projects
Project is a complete collection of repos, machines and services that can be interacted by an external user.
For instance, your online shop is a "project" 

Cluster is a set of machines, and node is a single machine (virtual or dedicated). 

Application is a piece of software that runs on a node.
There are subtypes of applications:
- services (`app.service`) which are expected to start accepting requests on start-up, and hang indefinitely
- pipelines (`app.pipeline`) which are launched by a trigger or schedule. They transform data, and halt.

```
Project: projectname
MyCluster: cluster(
    autoscale: ...
    nodenames: ...
    kubernetes_config_path: ..
    MyNode: node(
        hostname: ...
        region: ...
        bandwidth: ...
        ram: ...
        hdd: ...
        sdd: ...
        gpu: ...
        cpu: ...
        docker_image: ...
        dockerfile: ...
        MyApp: app(
            triggered: <schedule>
            >-- inName : mode[type] -->
            |>-- inName : mode[type] -->
            <-- outName: mode[type] --<
            |<-- outName: mode[type] --<
            |< apiCallName:   argmode[argtype] 
            |> apiCallName:   argmode[argtype]
            - - - > apiDefinition: argmode[argtype] < retval: argmode[argtype]
        )
    )
)
```

We do not have to define all these parts of hierarchy. "Default" project/cluster/node/app is 
assumed if not stated explicitly.