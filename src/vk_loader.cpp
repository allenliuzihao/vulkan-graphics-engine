
#include <vk_loader.h>

// load gltf meshes.
std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath) {
    std::cout << "Loading GLTF: " << filePath << std::endl;

    fastgltf::GltfDataBuffer data;
    // load gltf file
    data.loadFromFile(filePath);

    // specify gltf option.
    constexpr auto gltfOptions = fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;

    fastgltf::Asset gltf;
    fastgltf::Parser parser{};

    // parse loaded gltf, read in other gltf assets.
    auto load = parser.loadBinaryGLTF(&data, filePath.parent_path(), gltfOptions);
    if (load) {
        gltf = std::move(load.get());
    } else {
        fmt::print("Failed to load glTF: {} \n", fastgltf::to_underlying(load.error()));
        return {};
    }

    // as meshes are added, only pointers are recopoied, rather than meshes data.
    std::vector<std::shared_ptr<MeshAsset>> meshes;
    // use the same vectors for all meshes so that the memory doesnt reallocate as
    // often
    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;
    for (fastgltf::Mesh& mesh : gltf.meshes) {
        MeshAsset newmesh;
        // each mesh has a name.
        newmesh.name = mesh.name;

        // clear the mesh arrays each mesh, we dont want to merge them by error
        indices.clear();
        vertices.clear();

        // each mesh has a list of primitives, or submeshes.
        //  each submesh has a sub-region of indices as geometry data. 
        for (auto&& p : mesh.primitives) {
            // carve out a space in the final index buffer of where this submesh should reside in.
            GeoSurface newSurface;
            newSurface.startIndex = (uint32_t)indices.size();
            newSurface.count = (uint32_t)gltf.accessors[p.indicesAccessor.value()].count;

            // offset index by this for this submesh. 
            uint32_t initial_vtx = static_cast<uint32_t>(vertices.size());

            size_t numVertices = 0;

            // load indexes from gltf for this submesh.
            {
                fastgltf::Accessor& indexaccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexaccessor.count);

                // save vertex index into indices. 
                fastgltf::iterateAccessor<std::uint32_t>(gltf, indexaccessor,
                    [&](std::uint32_t idx) {
                        indices.push_back(idx + initial_vtx);
                    });
            }

            // load vertex positions
            {
                fastgltf::Accessor& posAccessor = gltf.accessors[p.findAttribute("POSITION")->second];
                numVertices = posAccessor.count;
                // add vertices positions.
                vertices.resize(vertices.size() + posAccessor.count);

                // fill up vertices array with additional position data from this sub-mesh. 
                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor,
                    [&](glm::vec3 v, size_t index) {
                        Vertex newvtx;
                        newvtx.position = v;
                        newvtx.normal = { 1, 0, 0 };
                        newvtx.color = glm::vec4{ 1.f };
                        newvtx.uv_x = 0;
                        newvtx.uv_y = 0;
                        vertices[initial_vtx + index] = newvtx;
                    });
            }

            // load vertex normals
            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end()) {
                auto& normalAccessor = gltf.accessors[normals->second];
                assert(numVertices == normalAccessor.count && "number vertices should be the same for vertex attributes.");

                // load up the normal data to vertices array for this submesh.
                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, normalAccessor,
                    [&](glm::vec3 v, size_t index) {
                        vertices[initial_vtx + index].normal = v;
                    });
            }

            // load UVs for this submesh.
            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end()) {
                auto& uvAccessor = gltf.accessors[uv->second];
                assert(numVertices == uvAccessor.count && "number vertices should be the same for vertex attributes.");

                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, uvAccessor,
                    [&](glm::vec2 v, size_t index) {
                        vertices[initial_vtx + index].uv_x = v.x;
                        vertices[initial_vtx + index].uv_y = v.y;
                    });
            }

            // load vertex colors for this submesh.
            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end()) {
                auto& colorAccessor = gltf.accessors[colors->second];
                assert(numVertices == colorAccessor.count && "number vertices should be the same for vertex attributes.");

                fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, colorAccessor,
                    [&](glm::vec4 v, size_t index) {
                        vertices[initial_vtx + index].color = v;
                    });
            }

            // add each submesh for this mesh.
            newmesh.surfaces.push_back(newSurface);
        }

        // display the vertex normals as color.
        constexpr bool OverrideColors = false;
        if (OverrideColors) {
            for (Vertex& vtx : vertices) {
                vtx.color = glm::vec4(vtx.normal, 1.f);
            }
        }

        // indices and vertices store data for all submeshes of this mesh.
        newmesh.meshBuffers = engine->uploadMesh(indices, vertices);
        
        // allocate in heap and then add to array of heap pointers for all meshes.
        meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newmesh)));
    }

    return meshes;
}